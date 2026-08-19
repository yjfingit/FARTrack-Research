from lib.test.evaluation.environment import EnvSettings_ITP


def local_env_settings():
    return EnvSettings_ITP(
        workspace_dir='/tmp/arbor-rgb-tracking-research-20260819',
        data_dir='/root/autodl-tmp/experiment/.research-assets/data',
        save_dir='/root/autodl-tmp/experiment/.research-assets/output',
    )
