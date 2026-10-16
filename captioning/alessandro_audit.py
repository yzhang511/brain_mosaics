"""Stage 1 dataset audit for Alessandro's t20 image-captioning sessions.

Reads only non-neural fields from the RedisMat .mat files (participant/session
metadata and the per-screen trial log); never loads binned_neural_*,
microphone_data or photodiode_data. Captions and image paths are the only
content of interest for the joint fine-tuning pilot; see
captioning/JOINT_FINETUNING_PLAN.md.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import scipy.io as sio

SESSIONS_ROOT = Path('/oak/stanford/groups/henderj/alessmv/data/t20/image_captioning')
BRAND_TASK_ASSETS = Path('/oak/stanford/groups/henderj/alessmv/repositories/brand-stanford/nodes/screens/task_assets')
RIG_TASK_ASSETS_PREFIX = '/home/nptl/brand/brand-modules/brand-stanford/nodes/screens/task_assets'

METADATA_FIELDS = ['participant', 'session_name', 'block_number', 'block_description', 'screens']
COCO_ID_RE = re.compile(r'(\d{12})\.jpg$')

COMPREHENSION_TASKS = {'NUM', 'ACT', 'LOC', 'CMP', 'EXP', 'COL', 'POS', 'PATH'}


def _val(x):
    """scipy returns empty char arrays for unset MATLAB string fields."""
    if hasattr(x, 'size') and x.size == 0:
        return ''
    return x


def resolve_image_path(raw_path):
    """Map the experiment rig's local path onto the mirrored Oak copy, if any."""
    if raw_path.startswith(RIG_TASK_ASSETS_PREFIX):
        relative = raw_path[len(RIG_TASK_ASSETS_PREFIX):].lstrip('/')
        return BRAND_TASK_ASSETS / relative
    return None


def classify_trial(task, cues, image_path):
    coco_match = COCO_ID_RE.search(image_path) if image_path else None
    if task == 'image_captioning' and coco_match:
        return 'coco_free_recall', coco_match.group(1)
    if task == 'do_nothing':
        return 'do_nothing', None
    joined = ' '.join(cues).lower()
    if 'type your sentence' in joined or 'original button' in joined:
        # Participant types/speaks their own sentence; no scripted caption is recorded.
        return 'free_typed_production', None
    if len(cues) == 2 and '?' in cues[0] and not cues[1].rstrip().endswith('?'):
        return 'templated_declarative', None
    if task in COMPREHENSION_TASKS:
        return 'comprehension_or_choice', None
    return 'unclassified', None


def iter_mat_files():
    return sorted(SESSIONS_ROOT.glob('t20.*/RedisMat/*.mat'))


def audit():
    records = []
    load_errors = []
    for mat_path in iter_mat_files():
        session = mat_path.parent.parent.name
        try:
            data = sio.loadmat(mat_path, simplify_cells=True, variable_names=METADATA_FIELDS)
        except Exception as error:
            load_errors.append({'file': str(mat_path), 'error': str(error)})
            continue
        screens = data.get('screens', [])
        if isinstance(screens, dict):
            screens = [screens]
        trials = defaultdict(list)
        for screen in screens:
            trial_number = screen.get('trial_number', '')
            if trial_number == '':
                continue
            trials[trial_number].append(screen)
        for trial_number, screen_list in trials.items():
            task = next((s.get('task') for s in screen_list if s.get('task')), '')
            raw_image = next((_val(s.get('image_path', '')) for s in screen_list if _val(s.get('image_path', ''))), '')
            cues = [_val(s.get('cue_text', '')) for s in screen_list if _val(s.get('cue_text', ''))]
            category, coco_id = classify_trial(task, cues, raw_image)
            resolved = resolve_image_path(raw_image) if raw_image else None
            records.append(dict(
                session=session,
                source_file=str(mat_path),
                trial_number=trial_number,
                task=task,
                category=category,
                coco_image_id=coco_id,
                raw_image_path=raw_image,
                resolved_image_path=str(resolved) if resolved else None,
                image_exists=bool(resolved and resolved.exists()),
                cues=cues,
                caption=cues[-1] if category == 'templated_declarative' else None,
            ))
    return records, load_errors


def summarize(records, load_errors):
    by_category = Counter(r['category'] for r in records)
    by_session_category = defaultdict(Counter)
    for r in records:
        by_session_category[r['session']][r['category']] += 1

    missing_images = [r for r in records if r['raw_image_path'] and not r['image_exists']]
    missing_by_prefix = Counter('/'.join(r['raw_image_path'].split('/')[:-1]) for r in missing_images)

    templated = [r for r in records if r['category'] == 'templated_declarative']
    caption_counts = Counter(r['caption'] for r in templated)
    duplicate_captions = {c: n for c, n in caption_counts.items() if n > 1}

    coco = [r for r in records if r['category'] == 'coco_free_recall']
    coco_image_counts = Counter(r['coco_image_id'] for r in coco)
    coco_with_image = sum(1 for r in coco if r['image_exists'])

    print(f'total mat files: {len(list(iter_mat_files()))}  (load errors: {len(load_errors)})')
    print(f'total trials parsed: {len(records)}')
    print()
    print('-- trials by category --')
    for cat, n in by_category.most_common():
        print(f'  {cat}: {n}')
    print()
    print('-- category by session --')
    for session in sorted(by_session_category):
        print(f'  {session}: {dict(by_session_category[session])}')
    print()
    print(f'-- templated_declarative (fixed 76-image set) -- n={len(templated)}')
    print(f'  unique captions: {len(caption_counts)}  duplicated captions: {len(duplicate_captions)}')
    print(f'  images resolved+exist on Oak: {sum(1 for r in templated if r["image_exists"])}/{len(templated)}')
    print()
    print(f'-- coco_free_recall (real COCO images, no scripted caption in .mat) -- n={len(coco)}')
    print(f'  distinct COCO image ids referenced: {len(coco_image_counts)}')
    print(f'  images resolved+exist on Oak: {coco_with_image}/{len(coco)}')
    print(f'  repeated image ids (shown >1 trial): {sum(1 for c in coco_image_counts.values() if c > 1)}')
    print()
    print(f'-- missing images (path recorded but not found on Oak) -- n={len(missing_images)}')
    for prefix, n in missing_by_prefix.most_common(10):
        print(f'  {n}  {prefix}')
    if load_errors:
        print()
        print('-- load errors --')
        for e in load_errors:
            print(' ', e)


if __name__ == '__main__':
    records, load_errors = audit()
    summarize(records, load_errors)
    out_path = Path(__file__).parent / 'alessandro_t20_manifest.json'
    out_path.write_text(json.dumps(records, indent=1))
    print()
    print(f'wrote {len(records)} trial records to {out_path}')
