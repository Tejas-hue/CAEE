CONTEXT AWARE EMPATHY ENGINE

Where machine learning meets emotional intelligence [finally]


>What This Actually Does

You’ve seen sentiment analysis before. It says:
“Angry.” “Happy.” “Sad.” Wow, thanks bot. Very helpful.


This goes further.
You give it a sentence → it tells you what the human being behind that sentence actually needs.
Support. Comfort. Validation. Motivation.
The soft stuff. The important stuff. The stuff most AI models politely ignore.


>Why I Built It (Instead of Sleeping)
Let me paint you a picture:
2 months ago, I was staring at Python like it was ancient Greek.
Now? I’ve duct-taped together a mini empathy engine using S-BERT embeddings, custom need-label mappings, and a tuned XGBoost classifier.

>Why?
Because emotion detection that stops at “anger” is like Google Maps saying “you’re lost.”
No solution. No direction. Saying someone’s “sad” doesn’t guide you — it just names the ache.

But what if, instead, we asked: What does this sadness long for? Maybe it’s comfort. Maybe it’s connection. Maybe it just wants to be seen;
Because behind every emotion is a human need, quietly hoping someone will notice.

So I built something that doesn’t just classify feelings — it contextualizes them.
Something closer to “Oh, you’re feeling hopeless? You probably need comfort.”
Not “ :) sad ”.

>Who Might Want This?
*Mental health startups who actually care about nuance
*Chatbot developers who want their bot to get IT
*Researchers who think emotions shouldn’t be a checkbox
*That one overly ambitious friend building an AI therapist instead of going to therapy (yes, you)

>Quick Demo
$ python scripts/predict_need.py
Enter a sentence: I feel left out and ignored.
→ support: 0.84  
→ validation: 0.12  
→ connection: 0.03
[Yes, it works. Yes, it’s slightly creepy.]

>What’s Under the Hood
I cobbled this together using:
*Python (obviously)
*GoEmotions from HuggingFace
*S-BERT for sentence embeddings
*XGBoost with Optuna for hyperparameter tuning (yes, I know what that means now)
*pandas, scikit-learn, joblib, and other tools blah blah blah

>Current Performance
Validation Accuracy: ~48%
(Considering the complexity of human emotions and how little training data maps to needs… doesn't seem half bad.)

Top labels like support, comfort, and validation are performing decently.
Others need work.

>Still To Come (a.k.a. The Vision Board)
*Drop XGBoost and move to a transformer-based classifier
*Add prediction explanations (like: why was “support” chosen?)
*Plug it into a live chatbot or journaling app
*Build an interactive demo in Streamlit (for non-dev humans)
*Create a feedback loop where the model learns what people actually need over time
Basically make it smarter than half the people I’ve met

>File Structure

Context-Aware Empathy Engine/
├── data/              # Cleaned & mapped dataset
├── models/            # Trained classifier + label encoder
├── scripts/           # Predict, train, prepare, rewrite, score
├── utils/             # Emotion → need mapping logic
├── notebooks/         # Where the chaos started
├── tests/             # Sanity-checks so I can sleep
├── outputs/           # Future visualizations live here
├── README.md          # You're reading it. Meta.

>About Me
Name’s Anant Pareek.

I study psychology. I make art. I write. I do loads of other tuff-stuff that gets my friends concerned.
And recently, I’ve been teaching myself machine learning — by building projects like this one that blend AI and emotion.
