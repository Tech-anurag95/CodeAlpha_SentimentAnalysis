"""
=============================================================================
CodeAlpha Data Analytics Internship
TASK 4: SENTIMENT ANALYSIS & NLP CUSTOMER INTELLIGENCE
Project: End-to-End Sentiment Classification & Aspect Mining for Customer Reviews
=============================================================================
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from wordcloud import WordCloud

# Ensure NLTK resources are available
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception:
    pass

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.autolayout'] = True

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VIS_DIR = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "food_delivery_reviews.csv")

def create_or_load_reviews():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)

    print("Synthesizing diverse customer review dataset...")
    reviews_data = [
        # Positive
        ("The food arrived piping hot and was absolutely delicious! Best pizza ever.", 5, "Food Quality", "Positive"),
        ("Super fast delivery, driver was very polite and professional. Highly recommended!", 5, "Delivery Speed", "Positive"),
        ("Fresh sushi, crisp packaging, and great portion size. Will order again!", 5, "Food Quality", "Positive"),
        ("Amazing burger and crispy fries. Delivery took only 20 minutes!", 5, "Delivery Speed", "Positive"),
        ("Excellent customer service resolved my item question instantly. 10/10 experience.", 5, "Customer Support", "Positive"),
        ("Koshary tasted authentic and fresh. Very affordable price and great value.", 4, "Pricing", "Positive"),
        ("The shawarma was packed with flavor and delivered warm. Love the garlic sauce!", 5, "Food Quality", "Positive"),
        ("Easy to order on the app, seamless credit card payment, on-time arrival.", 4, "App Experience", "Positive"),
        ("Very generous portions and neat eco-friendly packaging.", 5, "Packaging", "Positive"),
        ("Food was tasty, good temperature, and driver followed all delivery instructions.", 4, "Delivery Speed", "Positive"),
        ("Crispy fried chicken, not greasy at all. Excellent quality!", 5, "Food Quality", "Positive"),
        ("Great meal, fast courier, accurate order items.", 5, "Delivery Speed", "Positive"),
        ("Love the discount deals and quick checkout process.", 4, "Pricing", "Positive"),
        ("Fresh salad, dressing on the side as requested. Perfect lunch!", 5, "Food Quality", "Positive"),
        ("Driver called politely upon arrival and handed over the food carefully.", 5, "Delivery Speed", "Positive"),
        
        # Neutral
        ("Food was okay, nothing special but satisfied my hunger.", 3, "Food Quality", "Neutral"),
        ("Average delivery time, took about 40 minutes as estimated.", 3, "Delivery Speed", "Neutral"),
        ("The sandwich was fine, but a bit bland in terms of seasoning.", 3, "Food Quality", "Neutral"),
        ("Standard pizza, tasted decent but arrived slightly lukewarm.", 3, "Food Quality", "Neutral"),
        ("Portion was fair for the price, standard packaging.", 3, "Pricing", "Neutral"),
        ("Driver was fine, food arrived within the normal expected window.", 3, "Delivery Speed", "Neutral"),
        ("Average experience. Neither bad nor outstanding.", 3, "General", "Neutral"),
        ("The salad was decent, though dressing was slightly watery.", 3, "Food Quality", "Neutral"),
        
        # Negative
        ("Terrible experience! The food arrived completely cold after a 90 minute delay.", 1, "Delivery Speed", "Negative"),
        ("Food was spilled all over inside the plastic bag. Awful packaging!", 1, "Packaging", "Negative"),
        ("Wrong item delivered and the customer support was completely unhelpful.", 1, "Customer Support", "Negative"),
        ("The chicken was dry, burnt, and tasted stale. Extremely disappointed.", 1, "Food Quality", "Negative"),
        ("Way overpriced for tiny portions! Waste of money.", 1, "Pricing", "Negative"),
        ("The driver was rude, refused to come to the door, and got lost.", 1, "Delivery Speed", "Negative"),
        ("Missing drinks and extra sauces that I paid for. Horrible service!", 1, "Customer Support", "Negative"),
        ("App crashed during payment and charged me twice. Refund took days.", 1, "App Experience", "Negative"),
        ("Food smelled bad and looked undercooked. Could not eat it.", 1, "Food Quality", "Negative"),
        ("Late delivery by over an hour, driver was unapologetic.", 1, "Delivery Speed", "Negative"),
        ("Soggy fries and cold burger. Terrible quality control.", 2, "Food Quality", "Negative"),
        ("Support bot was useless and couldn't resolve my missing order issue.", 1, "Customer Support", "Negative")
    ]

    # Expand dataset with realistic variations to 1,500 samples
    np.random.seed(42)
    expanded = []
    for i in range(1500):
        base = reviews_data[np.random.randint(0, len(reviews_data))]
        review_text = base[0]
        rating = base[1]
        aspect = base[2]
        sentiment = base[3]
        
        # Add random modifier variations
        if sentiment == 'Positive' and np.random.rand() > 0.5:
            review_text += " " + np.random.choice(["Would definitely recommend!", "Super happy with this service.", "Five stars!"])
        elif sentiment == 'Negative' and np.random.rand() > 0.5:
            review_text += " " + np.random.choice(["Never ordering again!", "Total disaster.", "Please fix your service!"])

        expanded.append({
            'Review_ID': f"REV_{i+1:05d}",
            'Review_Text': review_text,
            'Rating': rating,
            'Aspect': aspect,
            'Sentiment': sentiment
        })

    df = pd.DataFrame(expanded)
    df.to_csv(DATA_FILE, index=False)
    print(f"Created & saved {len(df)} customer reviews to {DATA_FILE}")
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if w not in stop_words]
    except Exception:
        pass
    return " ".join(tokens)

def run_sentiment_pipeline(df):
    print("=" * 75)
    print("STEP 1: TEXT PREPROCESSING & VADER LEXICON SCORING")
    print("=" * 75)
    
    df['Cleaned_Text'] = df['Review_Text'].apply(clean_text)
    
    # VADER Sentiment Scoring
    sia = SentimentIntensityAnalyzer()
    df['Compound_Score'] = df['Review_Text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    df['Positive_Score'] = df['Review_Text'].apply(lambda x: sia.polarity_scores(str(x))['pos'])
    df['Negative_Score'] = df['Review_Text'].apply(lambda x: sia.polarity_scores(str(x))['neg'])
    df['Neutral_Score'] = df['Review_Text'].apply(lambda x: sia.polarity_scores(str(x))['neu'])

    def classify_vader(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['VADER_Sentiment'] = df['Compound_Score'].apply(classify_vader)
    print("VADER Sentiment Classification Preview:")
    print(df[['Review_Text', 'Compound_Score', 'VADER_Sentiment', 'Sentiment']].head())

    print("\n" + "=" * 75)
    print("STEP 2: MACHINE LEARNING CLASSIFICATION (TF-IDF + LOGISTIC REGRESSION)")
    print("=" * 75)
    
    X = df['Cleaned_Text']
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # Train Model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # -----------------------------------------------------------------------
    # Visualizations
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 3: GENERATING SENTIMENT VISUALIZATIONS & WORD CLOUDS")
    print("=" * 75)

    # 1. Sentiment Distribution & VADER Polarity Density
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sent_counts = df['Sentiment'].value_counts()
    axes[0].pie(sent_counts, labels=sent_counts.index, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c', '#95a5a6'],
                startangle=140, explode=(0.04, 0.04, 0.04), shadow=True)
    axes[0].set_title('Overall Sentiment Distribution', fontsize=13, fontweight='bold')

    sns.kdeplot(data=df, x='Compound_Score', hue='Sentiment', fill=True, palette={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}, ax=axes[1])
    axes[1].set_title('VADER Compound Polarity Score Density', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Polarity (-1.0 Negative to +1.0 Positive)')
    
    plt.tight_layout()
    p1 = os.path.join(VIS_DIR, "01_sentiment_distribution_and_polarity.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f" Saved: {p1}")

    # 2. Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    labels = sorted(df['Sentiment'].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f'Logistic Regression Confusion Matrix (Acc: {acc*100:.1f}%)', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('True Sentiment')
    plt.tight_layout()
    p2 = os.path.join(VIS_DIR, "02_ml_confusion_matrix.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f" Saved: {p2}")

    # 3. Aspect-Based Sentiment Breakdown
    plt.figure(figsize=(11, 5))
    sns.countplot(data=df, x='Aspect', hue='Sentiment', palette={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}, edgecolor='black')
    plt.title('Aspect-Based Sentiment: Food Quality, Delivery Speed, Support & Pricing', fontsize=13, fontweight='bold')
    plt.xlabel('Operational Aspect Category')
    plt.ylabel('Review Count')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    p3 = os.path.join(VIS_DIR, "03_aspect_sentiment_breakdown.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f" Saved: {p3}")

    # 4. Word Clouds: Positive vs Negative Feedback
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    pos_text = " ".join(df[df['Sentiment'] == 'Positive']['Cleaned_Text'])
    neg_text = " ".join(df[df['Sentiment'] == 'Negative']['Cleaned_Text'])

    wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(pos_text)
    wc_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(neg_text)

    axes[0].imshow(wc_pos, interpolation='bilinear')
    axes[0].set_title('Top Positive Sentiment Keywords', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(wc_neg, interpolation='bilinear')
    axes[1].set_title('Top Negative Sentiment Pain Points', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    p4 = os.path.join(VIS_DIR, "04_sentiment_wordclouds.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f" Saved: {p4}")

    return df

if __name__ == "__main__":
    df_reviews = create_or_load_reviews()
    run_sentiment_pipeline(df_reviews)
    print("\nTask 4 Sentiment Analysis Pipeline Completed Successfully!")
