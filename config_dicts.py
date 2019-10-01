def config_dict(parent_folder, model, gp_q, vae_q, dataset):
    config = {}

    if dataset == "mnist":
        config = {
            "output_distribution": "bernoulli",
            "batch_size": 500,
            "num_data_points": 60000,
            "state_size": [28, 28],
            "label_latent_manifold": True,
            "plot_all": False,
            "plot_dimensions": 2,
            # Number of inducing points for GP
            "num_ind_points_beta": 300,
            "num_ind_points_gamma": 800
        }
        # Training parameters
        if model == "vae_gplvm":
            config["learning_rate_kernels"] = 0.001
            config["learning_rate_global"] = 0.01
            config["learning_rate_mlp"] = 0.001
        # Training parameters
        if model == "vae_gplvm_regression":
            config["learning_rate_kernels"] = 0.001
            config["learning_rate_global"] = 0.01
            config["learning_rate_mlp"] = 0.001

    # Need to be updated
    # elif dataset == "mocap":
    #     config = {
    #         "output_distribution": "gaussian",
    #         "batch_size": 31,
    #         "num_data_points": 217,
    #         "state_size": [123],
    #         "label_latent_manifold": False,
    #         "plot_dimensions": 3,
    #         "learning_rate_local": 0.01,
    #         "learning_rate_global": 0.01
    #     }
    # elif dataset == "cmu_walk":
    #     config = {
    #         "output_distribution": "gaussian",
    #         "batch_size": 257,
    #         "num_data_points": 4369,
    #         "state_size": [123],
    #         "label_latent_manifold": False,
    #         "plot_dimensions": 3,
    #         "learning_rate_local": 0.01,
    #         "learning_rate_global": 0.01
    #     }
    elif dataset == "mixture_gauss":
        config = {
            "output_distribution": "gaussian",
            "batch_size": 50,
            "num_data_points": 1000,
            "state_size": [1],
            "label_latent_manifold": False,
            "plot_dimensions": 1,
            "plot_all": False,
            # Number of inducing points for GP
            "num_ind_points_beta": 30,
            "num_ind_points_gamma": 80,
        }
        # Training parameters
        if model == "vae_gplvm":
            config["learning_rate_kernels"] = 0.001
            config["learning_rate_global"] = 0.01
            config["learning_rate_mlp"] = 0.001
        # Training parameters
        if model == "vae_gplvm_regression":
            config["learning_rate_kernels"] = 0.001
            config["learning_rate_global"] = 0.01
            config["learning_rate_mlp"] = 0.001
    # elif dataset == "celebA":
    #     config = {
    #         "output_distribution": "bernoulli",
    #         "batch_size": 223,
    #         "num_data_points": 182637,
    #         "state_size": [64, 64],
    #         "label_latent_manifold": False,
    #         "plot_dimensions": 2,
    #         "learning_rate_local": 0.01,
    #         "learning_rate_global": 0.01
    #     }
    # elif dataset == "dual_moon":
    #     config = {
    #         "output_distribution": "gaussian",
    #         "batch_size": 50,
    #         "num_data_points": 500,
    #         "state_size": [2],
    #         "label_latent_manifold": False,
    #         "plot_all": True,
    #         "plot_dimensions": 2,
    #         # Training parameters
    #         "learning_rate_local": 0.01,
    #         "learning_rate_global": 0.01,
    #         "learning_rate_kernels": 0.01,
    #         # Number of inducing points for GP
    #         "num_ind_points_beta": 20,
    #         "num_ind_points_gamma": 60
    #     }
    # elif dataset == "spiral_3d":
    #     config = {
    #         # Data properties
    #         "output_distribution": "gaussian",
    #         "batch_size": 50,
    #         "num_data_points": 500,
    #         "state_size": [3],
    #         "label_latent_manifold": False,
    #         "plot_all": True,
    #         "plot_dimensions": 3,
    #         # Training parameters
    #         "learning_rate_local": 0.01,
    #         "learning_rate_global": 0.01,
    #         "learning_rate_kernels": 0.01,
    #         # Number of inducing points for GP
    #         "num_ind_points_beta": 20,
    #         "num_ind_points_gamma": 60
    #     }
    else:
        print("Invalid dataset")


    # Folders where everything is saved
    config["summary_dir"] = f"{parent_folder}/Z{vae_q}_X{gp_q}_summary/"
    config["results_dir"] = f"{parent_folder}/Z{vae_q}_X{gp_q}_results/"
    config["checkpoint_dir"] = f"{parent_folder}/Z{vae_q}_X{gp_q}_checkpoint/"
    # Max number of checkpoints to keep
    config["max_to_keep"] = 5
    # Latent dimensions
    config["gp_q"] = gp_q
    config["vae_q"] = vae_q
    if model == "vae_gplvm":
        # Training iterations
        config["global_iterations"] = 10
        config["pretraining_iterations"] = 2
        # Max training epochs
        config["num_epochs"] = 200
        config["epochs_mlp"] = 10
    else:
        config["global_iterations"] = 10
        config["num_epochs"] = 500
    # Number of draws used in marginal KL calculation and to get the test metrics
    config["num_draws"] = 240
    # Number of points in the X latent space from which to get reconstructed images
    config["num_plot_x_points"] = 30
    config["num_geodesics"] = 5
    # Max value in each dimension X latent space from which to generate images
    config["max_x_value"] = 2
    # Number of iterations per epoch
    config["num_iter_per_epoch"] = config["num_data_points"] // config["batch_size"]

    return config
