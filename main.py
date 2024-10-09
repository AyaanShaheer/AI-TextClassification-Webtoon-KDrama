#importing to be used libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


#dataset consist of title, description and thei categories (Anime/Kdrama)
data = {
    'title': [
        "Ouran High School Host Club",
        "Lookism",
        "Tomo-chan Is A Girl",
        "Blue Spring Ride (Ao Haru Ride)",
        "Romantic Killer",
        "Maid Sama!",
        "Skip Beat!!",
        "Horimiya",
        "kaguya-sama: Love Is War",
        "My Love Story!! (Ore Monogatari!!)",
        "Gangnam Beauty (2018)",
        "Lovely Runner (2024)",
        "The Beauty Inside (2018)",
        "18 Again (2020)",
        "The Interest of Love (2022-2023)"
    ],
    'description': [
        "A romantic comedy involving a high school host club.",
        "A story of a boy who can switch between two bodies with different looks.",
        "A romantic comedy about a tomboyish girl and her love interest.",
        "A story of a girl reuniting with her first love.",
        "A comedic story involving love and video games.",
        "A high school romance involving a girl working as a maid.",
        "A story about revenge in the entertainment industry.",
        "A romance between two seemingly different high school students.",
        "A romantic comedy with a strategic love battle between two students.",
        "A romantic comedy about an unlikely high school couple.",
        "A K-drama about a girl who undergoes plastic surgery to escape bullying.",
        "A romance involving competitive running and high school love.",
        "A love story where a person wakes up in a different body every day.",
        "A K-drama where a man reverts to his 18-year-old self to fix his life.",
        "A romantic K-drama involving complex love relationships in a workplace."
    ],
    'category': [
        'Anime', 'Anime', 'Anime', 'Anime', 'Anime',
        'Anime', 'Anime', 'Anime', 'Anime', 'Anime',
        'K-Drama', 'K-Drama', 'K-Drama', 'K-Drama', 'K-Drama'
    ]
}

#create a DataFrame
df = pd.DataFrame(data)

#combine title and description into a single feature for the model
df['combined_text'] = df['title'] + " " + df['description']


#Text processing using TF-IDF to convert text into numerical data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['combined_text'])

#targeting labels
y = df['category']

#splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#model logistical regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

#predicting on test data
log_pred = log_model.predict(X_test)

#evaluating model
print("Logistic Regression Accuracy: ", accuracy_score(y_test, log_pred))
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, log_pred))
