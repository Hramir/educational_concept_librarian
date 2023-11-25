import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os 
from evaluation_metrics.score_predictor import Regression_Model
from evaluation_metrics.score_predictor import NUM_FEATURES
from evaluation_metrics.score_predictor import NUM_FEATURES_WORD2VEC
from evaluation_metrics.score_predictor import BAR_WIDTH, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, TRANSCRIPT_DATA_PATH, TOPIC_COLUMNS
import pandas as pd

transcript_df = pd.read_csv(TRANSCRIPT_DATA_PATH)
view_count_labels = transcript_df['view_count']
score_labels = view_count_labels
TRANSCRIPT_INDICES_TO_SCORE_BRACKETS = dict()
median_score = score_labels.median()
for transcript_index, score in enumerate(score_labels):
    if score < median_score : TRANSCRIPT_INDICES_TO_SCORE_BRACKETS[transcript_index] = 0
    if median_score <= score : TRANSCRIPT_INDICES_TO_SCORE_BRACKETS[transcript_index] = 1
NUM_BRACKETS = 2
# number_quantiles = 4
# q0 = score_labels.quantile(1.0 / number_quantiles * 0)
# q1 = score_labels.quantile(1.0 / number_quantiles * 1)
# q2 = score_labels.quantile(1.0 / number_quantiles * 2)
# q3 = score_labels.quantile(1.0 / number_quantiles * 3)
# for transcript_index, score in enumerate(score_labels):
#     if q0 <= score < q1 : TRANSCRIPT_INDICES_TO_SCORE_BRACKETS[transcript_index] = 0
#     elif q1 <= score < q2 : TRANSCRIPT_INDICES_TO_SCORE_BRACKETS[transcript_index] = 1
#     elif q2 <= score < q3 : TRANSCRIPT_INDICES_TO_SCORE_BRACKETS[transcript_index] = 2
#     elif q3 <= score : TRANSCRIPT_INDICES_TO_SCORE_BRACKETS[transcript_index] = 3

class Score_Classifier(Regression_Model):
    def __init__(self, 
                date: str, 
                log_num: str, 
                type_of_classifier: str, 
                projection_type: str="SQ-R", 
                architecture: str="LAPS", 
                dataset: str="YouTube", 
                alpha=100):
        """
        1. Score Data YT, MITOCW
            Access Embedding Features

        2. SVM Classification
        
        3. Evaluate Classification Model: Accuracy, F1 Score, Precision, Recall
        
        4. Visualize Predicted score vs. Actual score
        """
        # def __init__(self, date : str, log_num : str, projection_type: str ="HR", dataset: str ="YouTub"):

        super().__init__(date, log_num, projection_type, dataset)
        type_of_classifier = type_of_classifier.lower()
        projection_type = projection_type.replace("-", "").upper()
        self.architecture = architecture
        self.dataset = dataset
        if type_of_classifier == "linear":
            self.classifier = SVC(kernel = "linear")
        elif type_of_classifier == "rbf":
            self.classifier = SVC(kernel = "rbf")
        elif type_of_classifier == "polynomial":
            raise AssertionError("Polynomial Classifier not implemented yet!")
            poly = PolynomialFeatures(degree=2)
            embeddings_poly = poly.fit_transform(embeddings)
            self.classifier = LinearRegression()
        elif type_of_classifier == "hyperbolic":
            raise AssertionError("Hyperbolic Classifier not implemented yet!")
            self.classifier = HyperbolicCentroidRegression()
        elif type_of_classifier == "random_forest":
            # self.classifier = RandomForestClassifier(n_estimators=200, random_state=21)
            self.classifier = RandomForestRegressor(n_estimators = 100, random_state=21)
        else:
            raise AssertionError(f"Invalid Classifier type : {type_of_classifier}!")
        self.train_score_brackets = [TRANSCRIPT_INDICES_TO_SCORE_BRACKETS[train_index] for train_index in self.train_indices]
        self.val_score_brackets = [TRANSCRIPT_INDICES_TO_SCORE_BRACKETS[val_index] for val_index in self.val_indices]
        self.test_score_brackets = [TRANSCRIPT_INDICES_TO_SCORE_BRACKETS[test_index] for test_index in self.test_indices]
        
        self.model_str = "SVC_Linear" if type_of_classifier == "linear" else "SVC_RBF" if type_of_classifier != "random_forest" else "Random_Forest"
        # self.projection_type = projection_type
        log_path = os.path.join("logs", "lp", date, log_num)
        self.embeddings_dir = os.path.join(log_path, 'embeddings')
    def classification(self) -> float:
        self.train()
        predicted_score_brackets, accuracy_score = self.test()
        print("Predicted Score Brackets : ", predicted_score_brackets)  
        # if self.model_str == "SVC_Linear": self.visualize_model_parameters(use_jet = False)

        self.plot_accuracy_per_score_bracket(predicted_score_brackets)
        
        
        # self.plot_score_brackets_vs_predicted_scores_brackets(predicted_score_brackets)
        # self.plot_score_brackets_vs_predicted_scores_brackets_curves(predicted_score_brackets)
        # TODO: Include F1 Score
        print(f"{self.model_str} Model with Projection {self.projection_type} Accuracy Score:", accuracy_score)
        print("THIS IS THE SCORE", accuracy_score)
        return accuracy_score
    
    def train(self):
        """
        1. Get embeddings
        2. Get score labels
        3. Perform classification
        4. Return predicted scores with score labels

        """
        train_embeddings_list = []
        val_embeddings_list = []
        test_embeddings_list = []
        topics_df = transcript_df[TOPIC_COLUMNS]

        topics_values = topics_df.values.tolist()
        topics_arrays = [np.array(row) for row in topics_values]

        for train_index in self.train_indices:
            train_embeddings = topics_arrays[train_index]            
            train_embeddings_list.append(train_embeddings)
        for val_index in self.val_indices:
            val_embeddings = topics_arrays[val_index]
            val_embeddings_list.append(val_embeddings)
        for test_index in self.test_indices:
            test_embeddings = topics_arrays[test_index]            
            test_embeddings_list.append(test_embeddings)
        
        train_embeddings_list += val_embeddings_list
        
        if type(self.classifier) == SVC or type(self.classifier) == RandomForestClassifier or type(self.classifier) == RandomForestRegressor:
            # Projection Mapping from 3D to 1D
            print("Projecting Train Embeddings :")
            projected_embeddings = self.project_embeddings(train_embeddings_list)
            if self.projection_type == "SQR": 
                scaler = StandardScaler()
                projected_embeddings = scaler.fit_transform(projected_embeddings)
            # self.regressor_model.fit(projected_embeddings, self.train_score_labels)
            self.classifier.fit(projected_embeddings, self.train_score_brackets + self.val_score_brackets)
        
    def test(self):
        """
        Must make sure training has been done beforehand
        Test predicted labels from embeddings
        Returns Predicted Classification Label, Accuracy between predicted labels and actual labels
        """
        
        val_embeddings_list = []
        test_embeddings_list = []
        
        topics_df = transcript_df[TOPIC_COLUMNS]

        topics_values = topics_df.values.tolist()
        topics_arrays = [np.array(row) for row in topics_values]
        for val_index in self.val_indices:
            val_embeddings = topics_arrays[val_index]
            val_embeddings_list.append(val_embeddings)
        for test_index in self.test_indices:
            test_embeddings = topics_arrays[test_index]            
            test_embeddings_list.append(test_embeddings)
        
        projected_embeddings = self.project_embeddings(test_embeddings_list)
        predicted_score_brackets = self.classifier.predict(projected_embeddings)
        print(f"PREDICTED score BRACKETS!!! : {predicted_score_brackets}")
        predicted_score_brackets = [round(predicted_score_bracket) for predicted_score_bracket in predicted_score_brackets]
        # Encode categorical labels to numerical values
        # label_encoder = LabelEncoder()
        # y_encoded = label_encoder.fit_transform(y)

        # # Split data into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # # Create and train a classifier
        # classifier = SVC(kernel='linear', decision_function_shape='ovr')
        # classifier.fit(X_train, y_train)

        # # Make predictions
        # y_pred = classifier.predict(X_test)

        # # Decode predicted labels
        # y_pred_decoded = label_encoder.inverse_transform(y_pred)

        # Calculate accuracy
        accuracy = accuracy_score(self.test_score_brackets, predicted_score_brackets)
        print("Accuracy:", accuracy)
        return predicted_score_brackets, accuracy
    def visualize_model_parameters(self, use_jet: bool=False):
        # plt.figure(figsize = (10, 10))
        plt.figure()
        plt.title(f"{self.model_str} Model with Projection {self.projection_type} Trained Parameters")
        plt.ylabel('Parameter Value')
        plt.xlabel('Topic Index')
        x = np.arange(NUM_FEATURES)
        if use_jet: 
            cmap = plt.cm.jet
            # plt.bar(x, self.classifier.coef_, color=cmap(x / len(x)))
        else:
            # support_vectors = self.classifier.support_vectors_
            # dual_coefficients = self.classifier.dual_coef_
            # print("Support Vectors:")
            # print(support_vectors)
            # print("\nDual Coefficients:")
            # print(dual_coefficients)
            # plt.bar(x, self.classifier.coef_)
            print(f"SVM CLassifier Weights Dimensions : {self.classifier.coef_.shape}")
            print(f"SVM CLassifier Weights: {self.classifier.coef_}")
    def plot_accuracy_per_score_bracket(self, predicted_score_brackets):
        accuracies_per_score_bracket = dict()
        for score_bracket in range(NUM_BRACKETS):
            score_bracket_indices = [index for index, test_score_bracket in enumerate(self.test_score_brackets) if test_score_bracket == score_bracket]
            score_bracket_specific_predictions = [predicted_score_brackets[score_bracket_index] for score_bracket_index in score_bracket_indices]
            accuracies_per_score_bracket[score_bracket] = accuracy_score([score_bracket] * len(score_bracket_indices), score_bracket_specific_predictions)
        plt.figure()
        # for index, (score_bracket, label_len) in enumerate(zip(score_brackets, label_lens)):
        # plt.bar(score_bracket, label_len, color = cmap(index / len(score_brackets)))
    
        cmap = plt.cm.jet
        plt.title(f"{self.model_str} Model with Projection {self.projection_type} Accuracy Per score Bracket")
        for index in range(NUM_BRACKETS):
            accuracies = list(accuracies_per_score_bracket.values())
            # plt.bar(index, accuracies[index], width = BAR_WIDTH, color = cmap(index / len(accuracies)))
            plt.bar(index, accuracies[index], width = BAR_WIDTH, color = "maroon" if index == 0 else "royalblue")

        plt.xticks([0, 1], ["Lower Metric Videos", "Higher Metric Videos"])
        plt.ylabel("Accuracy")
        plt.xlabel("Score Bracket")

    def plot_score_brackets_vs_predicted_scores_brackets(self, predicted_score_brackets):
        plt.figure()
        plt.title(f"{self.model_str} Model with Projection {self.projection_type} score Brackets vs Predicted score Brackets")
        plt.ylabel('Score Bracket')
        plt.xlabel('Transcript Index')

        # Plotting the barplots
        x = np.arange(len(self.test_indices))
        plt.bar(x - BAR_WIDTH/2, self.test_score_brackets, BAR_WIDTH, label="Ground Truth score Bracket")
        plt.bar(x + BAR_WIDTH/2, predicted_score_brackets, BAR_WIDTH, label='Predicted score Bracket')
        plt.legend()

    def plot_score_brackets_vs_predicted_scores_brackets_curves(self, predicted_score_brackets):
        plt.figure()
        plt.title(f"{self.model_str} Model with Projection {self.projection_type} score Brackets vs Predicted score Brackets")
        plt.ylabel('score Bracket')
        plt.xlabel('Transcript Index')

        # Plotting the barplots
        x = np.arange(len(self.test_indices))
        plt.plot(self.test_score_brackets, color="blue", label="Ground Truth score Bracket", marker="o", markersize=5)
        plt.plot(predicted_score_brackets, color="orange", label="Predicted score Bracket", marker="v", markersize=5)
        plt.legend()

