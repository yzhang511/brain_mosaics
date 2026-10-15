"""Optional W&B logging; local experiment artifacts remain authoritative."""
from pathlib import Path


class Tracker:
    def __init__(self, mode, project, entity, name, directory, config):
        self.run = None
        if mode != 'disabled':
            import wandb
            self.wandb = wandb
            self.run = wandb.init(project=project, entity=entity, name=name,
                dir=str(directory), mode=mode, config=config,
                settings=wandb.Settings(init_timeout=45))
            self.run.define_metric('train/step')
            self.run.define_metric('*', step_metric='train/step')

    def log(self, step, metrics):
        if self.run is not None:
            self.run.log({'train/step': step, **metrics})

    def samples(self, step, split, condition, samples):
        if self.run is not None:
            table = self.wandb.Table(columns=['image_id', 'reference', 'generated', 'exact_match'],
                data=[[s['image_id'], s['reference'], s['generated'], s['exact_token_match']] for s in samples])
            self.log(step, {f'{split}/{condition}/captions': table})

    def finish(self, summary, directory):
        if self.run is not None:
            self.run.summary.update(summary)
            # Small config/results files only; checkpoints remain local.
            for name in ('config.json', 'history.json', 'results.json'):
                self.run.save(str(Path(directory) / name), base_path=str(directory), policy='now')
            self.run.finish()
