import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import tkinter as tk
from tkinter import filedialog, messagebox
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data['comment'].fillna('', inplace=True)  # Ganti NaN dengan empty string
        data.columns = data.columns.str.strip()
        comments = data['comment']
        labels = data['label']
        return comments, labels
    except Exception as e:
        raise Exception(f"Failed to load data: {str(e)}")

def preprocess_and_train(data):
    comments, labels = data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(comments)
    y = labels
    
    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    return model, vectorizer

class SentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis with Decision Tree")
        
        self.load_training_button = tk.Button(root, text="Pilih Data Training", command=self.load_training_file)
        self.load_training_button.pack(pady=10)
        
        self.selected_train = tk.Label(root, text="", wraplength=400)
        self.selected_train.pack(pady=10)
        
        self.load_test_button = tk.Button(root, text="Pilih Data Pengujian", command=self.load_test_file)
        self.load_test_button.pack(pady=10)
        
        self.selected_test = tk.Label(root, text="", wraplength=400)
        self.selected_test.pack(pady=10)
        
        self.analyze_button = tk.Button(root, text="Analisis Sentimen", command=self.analyze_sentiment)
        self.analyze_button.pack(pady=10)
        
        self.result_label = tk.Label(root, text="", wraplength=400)
        self.result_label.pack(pady=10)
        
        self.visualize_button = tk.Button(root, text="Visualisasi WordCloud", command=self.visualize_wordcloud)
        self.visualize_button.pack(pady=10)
        
        self.training_data = None
        self.test_data = None
        self.model = None
        self.vectorizer = None
        self.test_comments = None
        self.predictions = None

    def load_training_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.training_data = load_data(file_path)
                self.model, self.vectorizer = preprocess_and_train(self.training_data)
                self.selected_train.config(text=f"File Training: {file_path}")
                messagebox.showinfo("File Loaded", "Training file loaded and model trained successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load training file: {str(e)}")
                
    def load_test_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.test_data = pd.read_csv(file_path)
                self.test_data['comment'].fillna('', inplace=True)
                self.selected_test.config(text=f"File Uji: {file_path}")
                messagebox.showinfo("File Loaded", "Test file loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load test file: {str(e)}")
                
    def analyze_sentiment(self):
        if self.test_data is not None and self.model is not None and self.vectorizer is not None:
            self.test_comments = self.test_data['comment']
            X_test = self.vectorizer.transform(self.test_comments)
            self.predictions = self.model.predict(X_test)
            self.test_data['predicted_label'] = self.predictions
            positive_count = sum((pred == 'positif' or pred == 'positive') for pred in self.predictions)
            negative_count = sum((pred == 'negatif' or pred == 'negative') for pred in self.predictions)
            
            result_text = f"Positive reviews: {positive_count}\nNegative reviews: {negative_count}"
            self.result_label.config(text=result_text)
        else:
            messagebox.showwarning("Missing Data", "Please load both training and test files first.")

    def visualize_wordcloud(self):
        if self.test_comments is not None and self.predictions is not None:
            positive_comments = " ".join(self.test_comments[self.predictions == 'positif']) + " " + " ".join(self.test_comments[self.predictions == 'positive'])
            negative_comments = " ".join(self.test_comments[self.predictions == 'negatif']) + " " + " ".join(self.test_comments[self.predictions == 'negative'])
            
            # Generate positive comments
            wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_comments)
            
            # Generate negative comments
            wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_comments)
            
            # Plotting WordClouds
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            ax1.imshow(wordcloud_positive, interpolation='bilinear')
            ax1.set_title('Positive Comments')
            ax1.axis('off')

            ax2.imshow(wordcloud_negative, interpolation='bilinear')
            ax2.set_title('Negative Comments')
            ax2.axis('off')

            # Display the plot in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.root)
            canvas.draw()
            canvas.get_tk_widget().pack()
        else:
            messagebox.showwarning("Missing Data", "Please analyze sentiment before visualizing WordCloud.")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Analisis Sentimen Ulasan Pengguna dengan Decision Tree")
    root.geometry("800x600") 
    app = SentimentAnalysisApp(root)
    root.mainloop()