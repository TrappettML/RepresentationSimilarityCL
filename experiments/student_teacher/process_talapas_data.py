from loss_plotter import generate_loss_plots, generate_diff_plot, generate_metric_plot

from pathlib import Path
from ipdb import set_trace




def get_files_and_paths(path_to_data: str)->list[str]:
    return [str(p) for p in Path(path_to_data).rglob('*') if not p.is_file()]

def make_training_plots(list_of_files, data_root, plot_root):
    result_paths = []
    for file in list_of_files:
        if 'hist' in file:
            continue
        loss_fig, diff_fig = generate_loss_plots(file, show_plots=False)
        comp_diff_fig = generate_diff_plot(file, show_plots=False)
        auc_fig = generate_metric_plot(file, show_plots=False)
        mean_fig = generate_metric_plot(file, show_plots=False, metric='mean')
        med_fig = generate_metric_plot(file, show_plots=False, metric='median')
        
        file_data_path = Path(file)
        rel_path = file_data_path.relative_to(Path(data_root)).with_suffix("")
        plot_path = Path(plot_root, rel_path)
        plot_path.mkdir(parents=True, exist_ok=True)
        # set_trace()
        loss_fig.write_html(f"{plot_path}/loss_fig.html")
        diff_fig.write_html(f"{plot_path}/diff_fig.html")
        comp_diff_fig.write_html(f"{plot_path}/comp_diff_fig.html")
        auc_fig.write_html(f"{plot_path}/auc_fig.html")
        auc_fig.write_image(f"{plot_path}/auc_fig.pdf")
        mean_fig.write_html(f"{plot_path}/mean_fig.html")
        med_fig.write_html(f"{plot_path}/median_fig.html")
        result_paths.append(plot_path)
    return result_paths

# get all the files
# save plots here: /home/users/MTrappett/mtrl/RepresentationSimilarityCL/experiments/student_teacher/data/result_figures/single_layers_runs

# then make more detailed plots

def main():
    # data_path = "/home/users/MTrappett/mtrl/RepresentationSimilarityCL/experiments/student_teacher/data/single_layer_runs"
    # plots_path = "/home/users/MTrappett/mtrl/RepresentationSimilarityCL/experiments/student_teacher/data/result_figures/single_layers_runs"
    data_path = "/home/users/MTrappett/mtrl/RepresentationSimilarityCL/loss_data/overlap/comparison"
    plots_path = data_path + "expert_test_plots/"
    list_of_files_paths = get_files_and_paths(data_path)
    make_training_plots(list_of_files_paths, data_path, plots_path)

    print(f"Finished with plots, plots located in {plots_path=}")
    # set_trace()

if __name__ == '__main__':
    main()