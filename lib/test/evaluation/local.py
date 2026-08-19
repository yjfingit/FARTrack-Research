import os

from lib.test.evaluation.environment import EnvSettings_ITP


def local_env_settings():
    settings = EnvSettings_ITP(
        workspace_dir='/tmp/arbor-rgb-tracking-research-20260819',
        data_dir='/root/autodl-tmp/experiment/.research-assets/data',
        save_dir='/root/autodl-tmp/experiment/.research-assets/output',
    )
    # The official LaSOT Testing archive is flat; use a separately generated
    # symlink-only index so the legacy loader's class/sequence layout leaves
    # archive contents untouched.
    settings.lasot_path = os.environ.get(
        'FARTRACK_LASOT_PATH',
        '/root/autodl-tmp/experiment/.research-assets/index/lasot_testing',
    )
    return settings
