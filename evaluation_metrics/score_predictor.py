# from nn_regressor import Neural_Network_Regressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA, NMF
from sklearn.manifold import TSNE, Isomap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
import numpy as np 
import networkx as nx
from typing import List
from typing import Tuple    
import time
from tqdm import tqdm
import numpy as np
import os 
import pickle

BAR_WIDTH = 0.35
NUM_FEATURES = 128 # TODO: Change to actual number of features depending on model feature dimensions
NUM_FEATURES_WORD2VEC = 64 # TODO: Change to actual number of features depending on model feature dimensions

score_labels = np.load(os.path.join("data_scraper", "score_labels_256.npy"))
class Regression_Model:
    def __init__(self, date : str, log_num : str, projection_type: str ="HR", dataset: str ="YouTube"):
        log_path = os.path.join("logs", "lp", date, log_num)
        print(f"Using Log Path : {log_path}")
        self.embeddings_dir = os.path.join(log_path, 'embeddings')
        projection_type = projection_type.replace("-", "").upper()
        self.projection_type = projection_type
        self.train_indices = self.get_split_indices("train")
        self.val_indices = self.get_split_indices("val")
        self.test_indices = self.get_split_indices("test")
        self.dataset = dataset
        if dataset not in ["YouTube", "MITOCW"]:
            raise ValueError("Dataset must be either YouTube or MITOCW")
        self.num_features = NUM_FEATURES if self.dataset == "YouTube" else NUM_FEATURES_WORD2VEC

    def project_embeddings(self, embeddings_list) -> np.ndarray:
        projection_function = self.get_projection_function()
        projected_embeddings = [projection_function(embeddings) for embeddings in tqdm(embeddings_list)]
        np_projected_embeddings = np.array(projected_embeddings)
        if self.projection_type != "HR": 
            projected_embeddings = np_projected_embeddings.reshape((len(np_projected_embeddings), self.num_features))
        return projected_embeddings
    # TODO: Figure out if should scale before or after projection
    def scale_embeddings(self, embeddings_list) -> np.ndarray:
        print("Scaling Embeddings :")
        scaler = StandardScaler()
        scaled_embeddings = [scaler.fit_transform(embeddings) for embeddings in tqdm(embeddings_list)]
        np_scaled_embeddings = np.array(scaled_embeddings)
        scaled_embeddings = np_scaled_embeddings.reshape((len(np_scaled_embeddings), self.num_features))
        return scaled_embeddings
    def get_projection_function(self):
        projection_function = lambda x : x
        if self.projection_type == "HR": 
            
            def inner_product(u, v):
                return -u[0]*v[0] + np.dot(u[1:], v[1:]) 
            def get_hyperbolic_radius(embeddings):
                origin = np.array([1, 0, 0]) # .to(self.args.device)
                return [np.arccosh(-1 * inner_product(origin, coord)) for coord in embeddings]
            
            # def get_squared_radius(embeddings):
            #     return [coord[0] ** 2 + coord[1] ** 2 + coord[2] ** 2 for coord in embeddings]
            projection_function = get_hyperbolic_radius

        elif self.projection_type == "TSNE": projection_function = TSNE(n_components=1, init='random', perplexity=3).fit_transform
        elif self.projection_type == "SVD": projection_function = TruncatedSVD(n_components=1).fit_transform
        elif self.projection_type == "PCA": projection_function = PCA(n_components=1).fit_transform
        elif self.projection_type == "ISOMAP": projection_function = Isomap(n_components=1).fit_transform
        else: raise AssertionError(f"Invalid Projection Type : {self.projection_type}!")
        # Other possibilities: MDS, LLE, Laplacian Eigenmaps, etc.
        return projection_function
            
    def get_split_indices(self, split_str):
        split_indices = []
        for split_embeddings_dir in os.listdir(self.embeddings_dir):
            if split_str not in split_embeddings_dir: continue
            _, _, split_index_str = split_embeddings_dir.split("_")
            split_index, _ = split_index_str.split(".")
            split_index = int(split_index)
            split_indices.append(split_index)
        return sorted(split_indices)

class Score_Predictor(Regression_Model):
    def __init__(self, date : str, log_num : str, type_of_regression: str, projection_type: str="HR", architecture: str="FHNN", dataset: str="YouTube", alpha=100):
        """
        1. Evaluate Regression Model: MSE
        
        2. Visualize Predicted Score vs. Actual Score
        """
        super().__init__(date, log_num, projection_type, dataset)
        type_of_regression = type_of_regression.lower()
        self.architecture = architecture
        self.dataset = dataset
        if type_of_regression == "linear":
            self.regressor_model = LinearRegression()
        elif type_of_regression == "ridge":
            print("Alpha Parameter :", alpha)
            self.regressor_model = Ridge(alpha = alpha)

        elif type_of_regression == "polynomial":
            raise AssertionError("Polynomial Regression not implemented yet!")
            poly = PolynomialFeatures(degree=2)
            embeddings_poly = poly.fit_transform(embeddings)
            self.regressor_model = LinearRegression()
        elif type_of_regression == "hyperbolic":
            raise AssertionError("Hyperbolic Regression not implemented yet!")
            self.regressor_model = HyperbolicCentroidRegression()
        elif type_of_regression == "random_forest":
            # self.regressor_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.regressor_model = RandomForestRegressor(n_estimators=100, random_state=42)
            # self.regressor_model = RandomForestRegressor(n_estimators=500, random_state=42)
        
        elif type_of_regression == "neural_network":
            input_size = NUM_FEATURES
            hidden_size = NUM_FEATURES
            output_size = 1
            lr = 0.001
            self.regressor_model = Neural_Network_Regressor(input_size, hidden_size, output_size, learning_rate=lr)
        else:
            raise AssertionError(f"Invalid Regression type : {type_of_regression}!")
        if dataset == "YouTube":
            self.train_score_labels = [score_labels[train_index] for train_index in self.train_indices] 
            self.val_score_labels = [score_labels[val_index] for val_index in self.val_indices]
            self.test_score_labels = [score_labels[test_index] for test_index in self.test_indices]
        elif dataset == "MITOCW":
            self.train_score_labels = [MITOCW_score_labels[train_index] for train_index in self.train_indices] 
            self.val_score_labels = [MITOCW_score_labels[val_index] for val_index in self.val_indices]
            self.test_score_labels = [MITOCW_score_labels[test_index] for test_index in self.test_indices]
        else:
            raise AssertionError(f"Invalid Dataset : {dataset}!")
        self.model_str = "Linear" if type(self.regressor_model) == LinearRegression \
            else "Ridge" if type(self.regressor_model) == Ridge \
            else "Random Forest" if type(self.regressor_model) == RandomForestRegressor \
            else "Feed-Forward NN" if type(self.regressor_model) == Neural_Network_Regressor \
            else "Unknown"
        
    def regression(self) -> float:
        self.train()
        predicted_scores, mse_score, correlation = self.test()
        # self.plot_score_labels_vs_predicted_scores(predicted_scores)
        self.visualize_model_parameters(use_jet = False)
        # TODO: Uncomment this
        # self.plot_difference_between_predicted_and_labels(predicted_scores)
        self.plot_score_labels_vs_predicted_scores_curves(predicted_scores)
        self.plot_score_labels_directly_to_predicted_scores_curves(predicted_scores)
        print(f"{self.model_str} Model with Projection {self.projection_type} Mean Squared Error (MSE):", mse_score)
        return mse_score, correlation
    def visualize_model_parameters(self, use_jet=False):
        # plt.figure(figsize = (10, 10))
        plt.figure()
        plt.title(f"{self.model_str} Model with Projection {self.projection_type} Trained Parameters")
        plt.ylabel('Parameter Value')
        plt.xlabel('Region Of Interest (ROI) Index')
        x = np.arange(self.num_features)
        cmap = plt.cm.jet        
        
        if type(self.regressor_model) == RandomForestRegressor:
            plt.bar(x, self.regressor_model.feature_importances_, color=cmap(x / len(x)))
            return 
        elif type(self.regressor_model) == Neural_Network_Regressor:
            print(f"Model Parameters : {[tensor.shape for tensor in self.regressor_model.parameters()]}")
            row_sums = [torch.sum(tensor, dim=0).detach().numpy() for tensor in self.regressor_model.parameters()]
            print("ROW SUMS : ", row_sums)
            x = np.arange(len(row_sums[0]))
            plt.bar(x, row_sums[0], color=cmap(x / len(x)))
            return
        
            # plt.imshow(self.regressor_model.parameters())
        
        if use_jet:
            plt.bar(x, self.regressor_model.coef_, color=cmap(x / len(x)))
        else:
            plt.bar(x, self.regressor_model.coef_)

    def plot_score_labels_vs_predicted_scores(self, predicted_scores):
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        # Set width of each bar

        # Plotting the barplots
        
        plt.bar(x - BAR_WIDTH/2, self.test_score_labels, BAR_WIDTH, label='score Label')
        plt.bar(x + BAR_WIDTH/2, predicted_scores, BAR_WIDTH, label='Predicted score')

        # Set labels, title, and legend
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Index')
        plt.ylabel('score')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type}')
        plt.legend()
        
        # Show the plot
        plt.show()
    
    def plot_score_labels_vs_predicted_scores_curves(self, predicted_scores):
        
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        # x = np.arange(len(self.test_indices + self.val_indices))
        plt.plot(x, predicted_scores, linestyle='-', marker='v', color='orange', label='Predicted score', markersize=5)
        # plt.plot(x, self.test_score_labels + self.val_score_labels, linestyle='-', marker='o', color='blue', label='score Label', markersize=5)
        plt.plot(x, self.test_score_labels, linestyle='-', marker='o', color='blue', label='score Label', markersize=5)
        # Set labels, title, and legend
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        # plt.xticks(x, self.test_indices + self.val_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Index')
        plt.ylabel('score')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type} Predicted scores')
        plt.ylim(0, 100)
        plt.legend()

        plt.show()
    
    def plot_score_labels_directly_to_predicted_scores_curves(self, predicted_scores):
        
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_score_labels))
        y = np.arange(len(predicted_scores))
        plt.scatter(self.test_score_labels, predicted_scores, c='blue', marker='o', label='Actual vs. Predicted')
        
        plt.xlabel('Subject score')
        plt.ylabel('Predicted score')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type} Predicted scores')
        plt.ylim(0, 100)
        
        # Add a diagonal line for reference (perfect prediction)
        plt.plot([min(self.test_score_labels), max(self.test_score_labels)], [min(self.test_score_labels), max(self.test_score_labels)], linestyle='--', color='gray', label='Perfect Prediction')

        plt.grid()
        plt.legend(loc='upper left')
        plt.show()        

    def plot_difference_between_predicted_and_labels(self, predicted_scores):
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        
        plt.bar(x - BAR_WIDTH/2, predicted_scores - self.test_score_labels, BAR_WIDTH, label='Predicted score - score Label')
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        # ax.set_xticklabels(self.test_indices, rotation=90)
        plt.xlabel('Subject Index')
        plt.ylabel('score')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type} Difference Plot')
        plt.legend()

        plt.show()

    
    def test(self) -> float:
        """
        Must make sure training has been done beforehand
        Test Predicted scores from embeddings
        Returns Predicted scores, MSE, and Correlation Coefficient between Predicted scores and Actual scores
        """
        
        test_embeddings_list = []
        val_embeddings_list = []
        for test_index in self.test_indices:
            test_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_test_{test_index}.npy'))
            test_embeddings_list.append(test_embeddings)
        for val_index in self.val_indices:
            val_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_val_{val_index}.npy'))
            val_embeddings_list.append(val_embeddings)
        
        if type(self.regressor_model) == LinearRegression \
            or type(self.regressor_model) == Ridge \
            or type(self.regressor_model) == RandomForestRegressor:
            print("Projecting Test Embeddings :")
            projected_embeddings = self.project_embeddings(test_embeddings_list)
            
            print("Scaling Projected Test Embeddings :")
            scaler = StandardScaler()
            projected_embeddings = scaler.fit_transform(projected_embeddings)
            predicted_scores = self.regressor_model.predict(projected_embeddings)
        if type(self.regressor_model) == Neural_Network_Regressor:
            print("Projecting Test Embeddings :")
            projected_embeddings = self.project_embeddings(test_embeddings_list)
            print("Scaling Projected Test Embeddings :")
            scaler = StandardScaler()
            projected_embeddings = scaler.fit_transform(projected_embeddings)
            projected_embeddings_tensor = torch.from_numpy(projected_embeddings).clone().detach().to(dtype=torch.float32).squeeze()
            predicted_scores = self.regressor_model.predict(projected_embeddings_tensor)

            predicted_scores = predicted_scores.squeeze() # Reduce extra dimension from [num_test, 1] to [num_test]
            predicted_scores = predicted_scores.detach().numpy()
        print("score Labels: ", self.test_score_labels)
        print("Predicted scores: ", predicted_scores)
        
        # TODO: RESTORE TO ONLY TEST score LABELS
        # return predicted_scores, \
        #         mean_squared_error(predicted_scores, self.test_score_labels + self.val_score_labels), \
        #         np.corrcoef(predicted_scores, self.test_score_labels + self.val_score_labels)[0, 1]
        return predicted_scores, \
                mean_squared_error(predicted_scores, self.test_score_labels), \
                np.corrcoef(predicted_scores, self.test_score_labels)[0, 1]

    def get_embeddings_to_labels(self):
        embeddings = []
        embeddings_directory = 'embeddings'
        embeddings_to_labels = dict()
        for embeddings_filename in os.listdir(embeddings_directory):
            if os.path.isfile(os.path.join(embeddings_directory, embeddings_filename)):
                _, _, train_index = embeddings_filename.split()
                train_index = int(train_index)
                
                score_label = score_labels[train_index]
                # MITOCW_score_label = MITOCW_score_labels[train_index]

                embeddings = np.load(os.path.join(embeddings_directory, embeddings_filename))
                # TODO: Matrix to Label seems inefficient
                embeddings_to_labels[tuple(embeddings)] = score_label
        return embeddings_to_labels
    
    def train(self) -> List[Tuple[float]]:
        """
        1. Get embeddings
        2. Get score labels
        3. Perform regression
        4. Return predicted scores with score labels

        """
        
        train_embeddings_list = []
        val_embeddings_list = []
        test_embeddings_list = []
        for train_index in self.train_indices:
            train_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_train_{train_index}.npy'))
            train_embeddings_list.append(train_embeddings)
        for val_index in self.val_indices:
            val_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_val_{val_index}.npy'))
            val_embeddings_list.append(val_embeddings)
        for test_index in self.test_indices:
            test_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_test_{test_index}.npy'))
            test_embeddings_list.append(test_embeddings)
        
        train_embeddings_list += val_embeddings_list
        # TODO: Change back to train + val
        # train_embeddings_list += test_embeddings_list
        if type(self.regressor_model) == LinearRegression \
            or type(self.regressor_model) == Ridge \
            or type(self.regressor_model) == RandomForestRegressor:
            # Projection Mapping from 3D to 1D
            print("Projecting Train Embeddings :")
            projected_embeddings = self.project_embeddings(train_embeddings_list)
            print("Scaling Projected Train Embeddings :")
            scaler = StandardScaler()
            projected_embeddings = scaler.fit_transform(projected_embeddings)
            # self.regressor_model.fit(projected_embeddings, self.train_score_labels)
            # TODO: Change back to train + val
            self.regressor_model.fit(projected_embeddings, self.train_score_labels + self.val_score_labels)
            # self.regressor_model.fit(projected_embeddings, self.train_score_labels + self.val_score_labels + self.test_score_labels)
        elif type(self.regressor_model == Neural_Network_Regressor):
            print("Projecting Train Embeddings :")
            projected_embeddings = self.project_embeddings(train_embeddings_list)
            print("Scaling Projected Train Embeddings :")
            scaler = StandardScaler()
            projected_embeddings = scaler.fit_transform(projected_embeddings)

            projected_embeddings_tensor = torch.from_numpy(np.array(projected_embeddings)) \
                .clone() \
                .detach() \
                .to(dtype=torch.float32) \
                .squeeze()
            score_labels_tensor = torch.from_numpy(np.array(self.train_score_labels + self.val_score_labels)) \
                .clone() \
                .detach() \
                .to(dtype=torch.float32) \
                .squeeze()
            # TODO: Change back to train + val
            # score_labels_tensor = torch.from_numpy(np.array(self.train_score_labels + self.val_score_labels + self.test_score_labels)) \
            #     .clone() \
            #     .detach() \
            #     .to(dtype=torch.float32) \
            #     .squeeze()
            self.regressor_model.train(projected_embeddings_tensor, score_labels_tensor)

        else: raise AssertionError("Invalid Regression Model!")

    def predict_score(self, embeddings) -> float:
        """
        Predict score from embeddings, no label difference calculation
        """
        if type(self.regressor_model) == LinearRegression:
            radii_sqs = [coord[0] ** 2 + coord[1] ** 2 + coord[2] ** 2 for coord in embeddings]
            predicted_score = self.regressor_model.predict(radii_sqs)
        return predicted_score

    def mse_loss(self, predicted_scores_with_score_labels) -> float:    
        """
        Return mean-squared errors between predicted and actual scores
        
        """
        # mse_loss = sum((predicted_score - score_label) ** 2 for predicted_score, score_label 
        #     in predicted_scores_with_score_labels) / len(predicted_scores_with_score_labels)
        predicted_scores = [pred_label[0] for pred_label in predicted_scores_with_score_labels]
        respective_score_labels = [pred_label[1] for pred_label in predicted_scores_with_score_labels]
        mse_loss = mean_squared_error(predicted_scores, respective_score_labels)
        return mse_loss



