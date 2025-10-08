from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.coesot_path = '/mnt/data/yjj/SpikeFET/data/COESOT'
    settings.fe108_path = '/home/work/yjj/FE108'
    settings.network_path = '/mnt/data/yjj/SpikeFET/output/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/mnt/data/yjj/SpikeFET'
    settings.result_plot_path = '/mnt/data/yjj/SpikeFET/output/test/result_plots'
    settings.results_path = '/mnt/data/yjj/SpikeFET/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/mnt/data/yjj/SpikeFET/output'
    settings.segmentation_path = '/mnt/data/yjj/SpikeFET/output/test/segmentation_results'
    settings.visevent_path = '/home/work/yjj/VisEvent'

    return settings

