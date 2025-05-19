import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class SegmentCaller:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.segments = []
        self.labels = []

    def fit(self, segments, labels):
        """
        Fit the KNN model using the provided segments and labels.
        
        :param segments: List of segment feature vectors.
        :param labels: List of labels corresponding to the segments.
        """
        self.segments = segments
        self.labels = labels
        self.knn.fit(segments, labels)

    def predict(self, context):
        """
        Predict the label of a music segment based on the provided context.
        
        :param context: Feature vector representing the context from the encoder.
        :return: Predicted label for the music segment.
        """
        return self.knn.predict([context])[0]

# Example usage (inwork)
if __name__ == "__main__":
    # Example segments and labels
    segments = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    labels = np.array(['classical', 'jazz', 'rock', 'pop'])

    # Example context from encoder
    context = np.array([0.35, 0.45])

    # Initialize and fit the SegmentCaller
    segment_caller = SegmentCaller(n_neighbors=3)
    segment_caller.fit(segments, labels)

    # Predict the label for the given context
    predicted_label = segment_caller.predict(context)
    print(f"Predicted label: {predicted_label}")