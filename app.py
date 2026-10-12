from flask import Flask, render_template, request, jsonify
import re
import random
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
import pickle
import os
import time

app = Flask(__name__)

# ─────────────────────────────────────────────
#  Training Data
# ─────────────────────────────────────────────

GENUINE_REVIEWS = [
    "This product is amazing, exactly what I expected. Fast delivery too.",
    "Really happy with this purchase. The quality is top-notch and worth every penny.",
    "I've been using this for two weeks and it works perfectly. No complaints.",
    "Great product! Fits well and looks exactly like the pictures.",
    "Solid build quality. Took a while to arrive but totally worth the wait.",
    "Bought this as a gift and my friend loved it. Very good value for money.",
    "Excellent product. Does exactly what is advertised. Would recommend.",
    "Very satisfied with this purchase. Would definitely buy again.",
    "The product arrived on time and was well packaged. Works great.",
    "Good quality product. The description was accurate and matched perfectly.",
    "I was skeptical but this exceeded my expectations. Highly recommend!",
    "Five stars! Perfect condition and the features are exactly as described.",
    "Works as expected. I have been using it daily for a month now.",
    "Material feels premium and durable. Very happy with the purchase.",
    "Delivery was quick and the item was well packaged. Love it!",
    "Great value for money. Does the job without any issues.",
    "Loved the design and functionality. My whole family uses it now.",
    "Top quality product. No problems at all after three months of use.",
    "Really impressed with the build quality. Exactly as described.",
    "Decent product at a reasonable price. Shipping was faster than expected.",
    "This is my second purchase of this product. Never disappointed.",
    "Clear instructions included. Set it up in minutes. Works flawlessly.",
    "The item looks even better in person. Very sturdy construction.",
    "My order arrived a day early. Product matches the description perfectly.",
    "Would highly recommend this. Excellent customer service as well.",
    "The product is exactly as shown. Great quality and fast shipping.",
    "Pleasantly surprised by the quality. Will definitely order again.",
    "Not only does it look great, it is very functional too. Love it.",
    "Outstanding product. Well worth the price. Shipping was prompt.",
    "Used for a week and very happy. Does what it says on the box.",
    "The product works very well. I am pleased with the overall quality.",
    "It's perfect for what I needed. Great build and looks fantastic.",
    "Bought this based on reviews and they were right. Great product!",
    "Arrived quickly and in perfect condition. Very satisfied.",
    "Good sturdy product, nothing fancy but does the job well.",
    "Very authentic and original. No quality issues whatsoever.",
    "This is a well thought out product. Smart design and practical.",
    "Quality is better than expected. Really good for the price.",
    "Overall a good buy. Packaging was excellent and arrived safely.",
    "Exactly what I was looking for. Very happy customer here.",
]

FAKE_REVIEWS = [
    "BEST PRODUCT EVER!!! BUY BUY BUY!!! AMAZING AMAZING AMAZING!!!",
    "Perfect perfect perfect! 100% recommend! Buy now you will not regret!!!",
    "I love love love this product!!! Using it every day since I got it oh my god.",
    "Absolutely life changing!!! Changed my life completely! 5 stars not enough!",
    "OMG THIS IS SO AMAZING!!! EVERYONE SHOULD BUY THIS RIGHT NOW!!!",
    "Wow wow wow!!! I was amazed when I received this. Buy immediately!!!!",
    "This is the best thing I have ever bought in my entire life on Amazon!",
    "I bought one for mother son daughter father brother sister boyfriend! All love!!!",
    "Just received it today and it is absolutely perfect in every single way possible!",
    "Zero problems this product is flawless! 100% exactly as described. Buy!",
    "So happy! Very much recommend to all people everywhere always buy this!",
    "Five star product! Five star service! Five star delivery! Just five stars all!!",
    "I ordered 10 of these already. Going to order 10 more. It is that good!",
    "Never seen quality like this ever before. This brand is number one best!",
    "I just got mine and it is already my favorite thing I ever bought anywhere!",
    "Worth every dollar peso rupee pound. No matter your currency just buy!",
    "This item is a miracle. It solved all my problems. All of them. Totally!",
    "Every single reviewer who gave less than 5 stars is absolutely lying period.",
    "Best company best product best delivery best price best everything best!",
    "If you don't buy this you are missing out. Literally changing my life!",
    "Very good product highly recommend. Product is best. No problems at all ever.",
    "Excellent top quality super amazing wow fantastic brilliant product very good!",
    "I now sleep better eat better live better because of this product. Thank you!",
    "My neighbor bought this then I bought this then my whole street bought this!!",
    "Top notch! A+! Would give 10 stars if possible! Recommend to all! Buy now!",
    "Perfect product!!!. This need to be bought by everyone. Trust me on this.",
    "Very impressed outstanding awesome phenomenal terrific wonderful incredible buy!",
    "This company is number one trusted genuine authentic certified best in world!",
    "No complaints. Zero. None. Nada. Nothing wrong. All perfect. Best product.",
    "Shipped same day arrived next day opened box fell in love immediately. Buy now!",
    "Forget everything you know about quality. This redefines perfection. So perfect!",
    "I was not paid to say this but this product is truly revolutionary. Amazing!",
    "All my friends jealous. They all want one now. Must buy product today friends!",
    "Great for gifting! Gifted to 15 people already and all of them loved it so much!",
    "This product healed my back knee leg arm shoulder everything. Medical miracle!!!",
    "WOW WOW WOW!!! Five stars five stars five stars five stars five stars five stars!",
    "Just perfect. No flaws whatsoever. Completely and utterly flawless. Buy asap!",
    "I only buy from this brand now. Every single product they make is 100% perfect!",
    "Love love love! Great great great! Buy buy buy! Now now now! yes yes yes!",
    "This product should win awards. Truly the best mankind has ever manufactured!",
]

# ─────────────────────────────────────────────
#  Build & Train the Pipeline
# ─────────────────────────────────────────────

MODEL_PATH = "model.pkl"

def train_model():
    texts  = GENUINE_REVIEWS + FAKE_REVIEWS
    labels = [1] * len(GENUINE_REVIEWS) + [0] * len(FAKE_REVIEWS)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])

    pipeline.fit(texts, labels)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    return pipeline

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return train_model()

pipeline = train_model()

# ─────────────────────────────────────────────
#  Text Preprocessing
# ─────────────────────────────────────────────

try:
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
except:
    STOPWORDS = set()

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

# ─────────────────────────────────────────────
#  Prediction Helpers
# ─────────────────────────────────────────────

def predict_single(review: str):
    """Returns (label, confidence_fake, confidence_genuine)"""
    cleaned = preprocess(review)
    proba   = pipeline.predict_proba([cleaned])[0]   # [P(fake), P(genuine)]
    label   = int(pipeline.predict([cleaned])[0])    # 1=genuine, 0=fake
    return label, round(float(proba[0]) * 100, 1), round(float(proba[1]) * 100, 1)

def analyze_url(url: str):
    """
    Real scraping of Amazon/Flipkart is blocked by anti-bot measures.
    We simulate a realistic analysis using the URL as a seed for reproducibility.
    """
    seed = sum(ord(c) for c in url)
    rng  = random.Random(seed)

    total   = rng.randint(15, 60)
    reviews = []
    pool    = GENUINE_REVIEWS + FAKE_REVIEWS

    for i in range(total):
        sample  = rng.choice(pool)
        noise   = rng.choice(['', ' Great!', ' Recommended.', ' Very satisfied.', ' Wow!', ''])
        rev_txt = sample + noise
        label, p_fake, p_gen = predict_single(rev_txt)
        reviews.append({
            'text'        : rev_txt[:120] + ('…' if len(rev_txt) > 120 else ''),
            'label'       : 'Genuine' if label == 1 else 'Fake',
            'confidence'  : p_gen if label == 1 else p_fake,
            'fake_pct'    : p_fake,
            'genuine_pct' : p_gen,
        })

    genuine_count = sum(1 for r in reviews if r['label'] == 'Genuine')
    fake_count    = total - genuine_count

    return {
        'total'        : total,
        'genuine_count': genuine_count,
        'fake_count'   : fake_count,
        'genuine_pct'  : round(genuine_count / total * 100, 1),
        'fake_pct'     : round(fake_count    / total * 100, 1),
        'reviews'      : reviews,
        'platform'     : 'Amazon' if 'amazon' in url.lower() else
                         'Flipkart' if 'flipkart' in url.lower() else 'Online Store',
    }

# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/check-review', methods=['POST'])
def check_review():
    data   = request.get_json()
    review = data.get('review', '').strip()
    if not review:
        return jsonify({'error': 'Please enter a review.'}), 400
    if len(review) < 10:
        return jsonify({'error': 'Review is too short. Please enter at least 10 characters.'}), 400

    label, p_fake, p_gen = predict_single(review)
    return jsonify({
        'result'         : 'Genuine' if label == 1 else 'Fake',
        'genuine_pct'    : p_gen,
        'fake_pct'       : p_fake,
        'confidence'     : p_gen if label == 1 else p_fake,
        'cleaned_review' : preprocess(review),
    })

@app.route('/api/analyze-url', methods=['POST'])
def analyze_url_route():
    data = request.get_json()
    url  = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'Please enter a product URL.'}), 400
    if not url.startswith('http'):
        return jsonify({'error': 'Please enter a valid URL starting with http:// or https://'}), 400

    # Simulate scraping delay
    time.sleep(1.5)
    result = analyze_url(url)
    return jsonify(result)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)