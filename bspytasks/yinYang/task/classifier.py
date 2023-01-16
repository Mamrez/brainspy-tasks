from brainspy.utils.pytorch import TorchUtils
import os
import torch
import pickle as p
import matplotlib.pyplot as plt

from brainspy.utils.io import create_directory, create_directory_timestamp
from bspytasks.utils.io import save
from brainspy.utils.manager import get_optimizer

from bspytasks.yinYang.data import YinYangDataset


# These two lines should be added for other bspy tasks
from brainspy.utils.performance.accuracy import (get_accuracy, plot_perceptron)
from brainspy.utils.signal import pearsons_correlation


def yinYang_task(
    configs,
    dataloaders,
    custom_model,
    criterion,
    algorithm,
    logger=None,
    is_main=True,
    save_data=True,
):
    results = {}
    results["gap"] = str(configs["data"]["gap"])
    print(
        "=========================================================================================="
    )
    print("GAP: " + str(results["gap"]))

    results_dir, reproducibility_dir = init_dirs(
        str(results["gap"]),
        configs["results_dir"],
        is_main=is_main,
        save_data=save_data,
    )

    # criterion = get_criterion(configs['algorithm'])
    model = custom_model(configs['processor'])
    model = TorchUtils.format(model)    
    
    optimizer = get_optimizer(model, configs["algorithm"])

    # algorithm = get_algorithm(configs['algorithm'])
    model, train_data = algorithm(
        model,
        [dataloaders[0], dataloaders[1]],
        criterion,
        optimizer,
        configs["algorithm"],
        # logger=logger,
        save_dir=reproducibility_dir,
    )

    results["train_results"] = postprocess(
        configs["accuracy"],
        dataloaders[0].dataset,  # [dataloaders[0].sampler.indices],
        model,
        criterion,
        logger,
        save_dir=results_dir,
        name="train",
    )

    results["train_results"]["performance_history"] = train_data[
        "performance_history"][0]

    if len(dataloaders[1]) > 0:
        results["dev_results"] = postprocess(
            configs["accuracy"],
            dataloaders[1].dataset,  # [dataloaders[1].sampler.indices],
            model,
            criterion,
            logger,
            node=results["train_results"]["accuracy"]["node"],
            save_dir=results_dir,
            name="validation",
        )
        
        results["dev_results"]["performance_history"] = train_data[
            "performance_history"][1]

    if len(dataloaders[2]) > 0:
        results["test_results"] = postprocess(
            configs["accuracy"],
            dataloaders[2].dataset,  # [dataloaders[2].sampler.indices],
            model,
            criterion,
            logger,
            node=results["train_results"]["accuracy"]["node"],
            save_dir=results_dir,
            name="test",
        )

    if save_data:
        close(model, results, configs, reproducibility_dir, results_dir)

    print(
        "=========================================================================================="
    )

    return results, model


def close(model, results, configs, reproducibility_dir, results_dir):
    save("configs",
         os.path.join(reproducibility_dir, "configs.yaml"),
         data=configs)
    torch.save(
        results,
        os.path.join(reproducibility_dir, "results.pickle"),
        pickle_protocol=p.HIGHEST_PROTOCOL,
    )
    plot_results(results, plots_dir=results_dir)
    if model.is_hardware():
        model.load_state_dict(
            torch.load(
                os.path.join(reproducibility_dir,
                             "training_data.pickle"))['model_state_dict'])
    else:
        if configs['algorithm']['return_best_model']:
            model = torch.load(
                os.path.join(reproducibility_dir, "best_model_raw.pt"))
        else:
            # Return model as it was in the last epoch
            model = torch.load(
                os.path.join(reproducibility_dir, "model_raw.pt"))
    if model.is_hardware() and "close" in dir(model):
        model.close()

def get_yinYang_data():

    batch_size = 20

    dataset_train = YinYangDataset(size=10000, seed=42)
    dataset_validation = YinYangDataset(size=1000, seed=41)
    dataset_test = YinYangDataset(size=1000, seed=40)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=20)
    val_loader = torch.utils.data.DataLoader(dataset_validation)
    test_loader =torch.utils.data. DataLoader(dataset_test)

    return [train_loader, val_loader, test_loader]

def postprocess(configs,
                dataset,
                model,
                criterion,
                logger,
                node=None,
                save_dir=None,
                name="train"):
    results = {}
    with torch.no_grad():
        model.eval()
        inputs, targets = dataset[:]
        inputs = TorchUtils.format(inputs)
        targets = TorchUtils.format(targets)
        indices = torch.argsort(targets[:, 0], dim=0)
        inputs, targets = inputs[indices], targets[indices]
        predictions = model(inputs)
        results["performance"] = criterion(predictions, targets)

    # results['gap'] = dataset.gap
    results["inputs"] = inputs
    results["targets"] = targets
    results["best_output"] = predictions
    results["accuracy"] = get_accuracy(
        predictions, targets, configs, node=node
    )  # accuracy(predictions.squeeze(), targets.squeeze(), plot=None, return_node=True)
    print(
        f"{name.capitalize()} accuracy: {results['accuracy']['accuracy_value']}"
    )
    results["correlation"] = pearsons_correlation(predictions, targets)
    # results['accuracy_fig'] = plot_perceptron(results['accuracy'], save_dir, name=name)

    return results


def init_dirs(gap, base_dir, is_main=False, save_data=False):
    main_dir = "ring_classification_gap_" + gap
    reproducibility_dir = "reproducibility"
    results_dir = "results"
    if is_main:
        base_dir = create_directory_timestamp(base_dir, main_dir)
    if save_data:
        reproducibility_dir = os.path.join(base_dir, reproducibility_dir)
    else:
        reproducibility_dir = os.path.join(base_dir, reproducibility_dir,
                                           "tmp")
    create_directory(reproducibility_dir)
    results_dir = os.path.join(base_dir, results_dir)
    create_directory(results_dir)
    return results_dir, reproducibility_dir


def plot_results(results, plots_dir=None, show_plots=False, extension="png"):
    plot_output(results["train_results"],
                "Train",
                plots_dir=plots_dir,
                extension=extension)
    plot_perceptron(results["train_results"]["accuracy"],
                    plots_dir,
                    name="train")
    if "dev_results" in results:
        plot_output(results["dev_results"],
                    "Dev",
                    plots_dir=plots_dir,
                    extension=extension)
        plot_perceptron(results["dev_results"]["accuracy"],
                        plots_dir,
                        name="dev")
    if "test_results" in results:
        plot_output(results["test_results"],
                    "Test",
                    plots_dir=plots_dir,
                    extension=extension)
        plot_perceptron(results["test_results"]["accuracy"],
                        plots_dir,
                        name="test")
    plt.figure()
    plt.title(f"Learning profile", fontsize=12)
    plt.plot(
        TorchUtils.to_numpy(results["train_results"]["performance_history"]),
        label="Train",
    )
    if "dev_results" in results:
        plt.plot(
            TorchUtils.to_numpy(results["dev_results"]["performance_history"]),
            label="Dev",
        )
    plt.legend()
    if plots_dir is not None:
        plt.savefig(os.path.join(plots_dir, f"training_profile." + extension))

    plt.figure()
    plt.title(f"Inputs (V) \n {results['gap']} gap (-1 to 1 scale)",
              fontsize=12)
    plot_inputs(results["train_results"], "Train")
    if "dev_results" in results:
        plot_inputs(results["dev_results"], "Dev")
    if "test_results" in results:
        plot_inputs(results["test_results"], "Test")
    plt.legend()
    # if type(results['dev_inputs']) is torch.Tensor:
    if plots_dir is not None:
        plt.savefig(os.path.join(plots_dir, f"input." + extension))

    if show_plots:
        plt.show()
    plt.close("all")


def plot_output(results, label, plots_dir=None, extension="png"):
    plt.figure()
    plt.plot(results["best_output"].detach().cpu())
    plt.title(
        f"{label} Output (nA) \n Performance: {results['performance']} \n Accuracy: {results['accuracy']['accuracy_value']}",
        fontsize=12,
    )
    if plots_dir is not None:
        plt.savefig(os.path.join(plots_dir, label + "_output." + extension))


def plot_inputs(results,
                label,
                colors=["b", "r", "g"],
                plots_dir=None,
                extension="png"):
    # if type(results['dev_inputs']) is torch.Tensor:
    inputs = results["inputs"].cpu().numpy()
    targets = results["targets"][:, 0].cpu().numpy()
    # else:
    #     inputs = results['dev_inputs']
    #     targets = results['dev_targets']
    plt.scatter(
        inputs[targets == 0][:, 0],
        inputs[targets == 0][:, 1],
        marker=".",
        c=colors[0],
        label="Class 0 (" + label + ")",
        cmap=colors[0],
    )
    plt.scatter(
        inputs[targets == 1][:, 0],
        inputs[targets == 1][:, 1],
        marker="x",
        c=colors[1],
        label="Class 1 (" + label + ")",
        cmap=colors[1],
    )
    plt.scatter(
        inputs[targets == 2][:, 0],
        inputs[targets == 2][:, 1],
        marker="o",
        label = "Class 2 (" + label + ")",
        cmap=colors[2]
    )


if __name__ == "__main__":
    from torchvision import transforms as tfms

    from brainspy.utils import manager
    from brainspy.utils.io import load_configs
    from bspytasks.utils.transforms import DataToTensor

    from bspytasks.models.default_yinYang import SingleDNPUCustomModel as DefaultCustomModel
    # from bspytasks.models.default_yinYang import DefaultCustomModel

    configs = load_configs("configs/yinYang.yaml")

    criterion = manager.get_criterion(configs["algorithm"]['criterion'])
    algorithm = manager.get_algorithm(configs["algorithm"]['type'])

    dataloaders = get_yinYang_data()

    yinYang_task(configs, dataloaders, DefaultCustomModel, criterion, algorithm)

    print("")
