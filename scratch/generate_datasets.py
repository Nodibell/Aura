import os
import numpy as np
import pandas as pd

def generate_iris():
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    df['species'] = [iris.target_names[i] for i in iris.target]
    df.to_csv('sample_data/iris.csv', index=False)
    print("Generated iris.csv with", len(df), "rows.")

def generate_house_prices():
    np.random.seed(42)
    n = 500
    
    sqft = np.random.normal(2000, 600, n).astype(int)
    sqft = np.clip(sqft, 800, 5000)
    
    # bedrooms depends on sqft
    bedrooms = (sqft / 600).astype(int) + np.random.randint(-1, 2, n)
    bedrooms = np.clip(bedrooms, 1, 6)
    
    # bathrooms depends on bedrooms
    bathrooms = (bedrooms * 0.75 + np.random.choice([0.0, 0.5, 1.0], n, p=[0.2, 0.6, 0.2])).astype(float)
    bathrooms = np.clip(bathrooms, 1.0, 5.0)
    
    age = np.random.randint(0, 80, n)
    
    neighborhoods = np.random.choice(['Suburban', 'Urban', 'Rural'], n, p=[0.45, 0.35, 0.20])
    
    # Binary flag (pool) - more likely for larger houses
    has_pool = []
    for s in sqft:
        prob = 0.05 if s < 1800 else (0.2 if s < 2800 else 0.5)
        has_pool.append(int(np.random.rand() < prob))
    has_pool = np.array(has_pool)
    
    # Price model
    price = 50000 + sqft * 150 + bedrooms * 20000 + bathrooms * 15000 - age * 1200 + has_pool * 40000
    # Add noise
    price += np.random.normal(0, 25000, n)
    price = np.clip(price, 80000, None).astype(int)
    
    # Add a few outliers (e.g. extremely high price)
    price[np.random.choice(n, 5, replace=False)] = price[np.random.choice(n, 5, replace=False)] * 2.2
    
    # Create DataFrame
    df = pd.DataFrame({
        'SquareFootage': sqft,
        'Bedrooms': bedrooms.astype(float),  # float to support NaNs
        'Bathrooms': bathrooms,
        'Age': age.astype(float),            # float to support NaNs
        'Neighborhood': neighborhoods,
        'HasPool': has_pool,
        'Price': price
    })
    
    # Introduce missing values (5% in Bedrooms and Age)
    mask_beds = np.random.rand(n) < 0.05
    mask_age = np.random.rand(n) < 0.05
    df.loc[mask_beds, 'Bedrooms'] = np.nan
    df.loc[mask_age, 'Age'] = np.nan
    
    df.to_csv('sample_data/house_prices.csv', index=False)
    print("Generated house_prices.csv with", len(df), "rows.")

def generate_movie_reviews():
    pos_reviews = [
        "I absolutely loved this movie! The acting was superb and the story was incredibly moving. Highly recommended.",
        "An absolute masterpiece of modern cinema. Visuals were breathtaking and the performances were top notch.",
        "Really enjoyed it. Solid characters and a great soundtrack. Worth watching.",
        "Great direction and writing, but some scenes felt a bit too long. Overall a good experience.",
        "Brilliant cinematography and a powerful message. It kept me on the edge of my seat.",
        "Decent family movie with good humor. Nothing groundbreaking, but fun.",
        "Fantastic performances by the lead actors. The script was smart and witty.",
        "A beautiful story about love and loss. It made me cry. Truly outstanding.",
        "A fun ride from start to finish. Lots of action and a surprising twist.",
        "A heartwarming and delightful film that will leave you smiling.",
        "Surprisingly good. I went in with low expectations but was thoroughly entertained.",
        "An incredible journey. The acting was emotional and highly realistic.",
        "Excellent documentary. Very informative and kept me engaged throughout.",
        "A solid thriller with strong tension and great acting. Recommended.",
        "Loved the atmosphere and style of this film. A unique and memorable watch.",
        "A masterclass in storytelling. The dialogue is sharp and the pacing is perfect.",
        "One of the best movies of the year. Captivating from the very first frame to the last.",
        "Incredible cinematography and sound design. It's a feast for the senses.",
        "Highly engaging. The plot is filled with suspense and intelligence.",
        "A touching and authentic portrayal. A must-see for fans of the genre."
    ]
    
    neg_reviews = [
        "This was one of the worst films I have ever seen. Boring plot, terrible acting, and complete waste of time.",
        "I fell asleep halfway through. Very slow paced and nothing interesting happens.",
        "Horrible. The characters were annoying and the dialogue was poorly written.",
        "A complete disaster. I don't understand how this got greenlit. Avoid at all costs.",
        "Very disappointing. The trailer made it look much better than it actually was.",
        "The plot made absolutely no sense. The CGI was cheap and distracting.",
        "Stupid, loud, and annoying. Save your money and skip this one.",
        "Terrible pacing and generic characters. I expected much more from this director.",
        "Waste of time and talent. Everyone involved in this should be ashamed.",
        "Uninspirational and lazy writing. A major step down for the franchise.",
        "Awful. Zero plot, flat acting, and annoying sound design.",
        "This film has absolutely no redeeming qualities. It is painful to watch.",
        "Predictable story, mediocre acting, and a disappointing ending.",
        "Cheap jumpscares and awful writing. Not scary, just frustrating.",
        "A boring mess. Nothing works, from the dialogue to the soundtrack.",
        "Utterly pointless and exhausting. I wanted to walk out of the theater.",
        "Terrible script and uninspired performances. Don't waste your time.",
        "It was painful to sit through. Avoid it at all cost.",
        "Flat, lifeless, and completely devoid of any real emotion or drama.",
        "An embarrassment to filmmaking. A chaotic and unwatchable disaster."
    ]
    
    np.random.seed(42)
    n = 200
    reviews = []
    sentiments = []
    
    for _ in range(n):
        is_pos = np.random.rand() > 0.5
        sent = "positive" if is_pos else "negative"
        pool = pos_reviews if is_pos else neg_reviews
        text = np.random.choice(pool)
        
        # Add slight variation/noise
        fillers = [
            "", " Honestly, I think so.", " Truly a generic experience.", " Just my opinion.",
            " Some parts were okay though.", " I watched it last night.", " Definitely check it out.",
            " What a shame.", " I'm glad I saw it.", " Disappointed me a lot."
        ]
        text += np.random.choice(fillers)
        
        reviews.append(text)
        sentiments.append(sent)
        
    df = pd.DataFrame({
        'Review': reviews,
        'Sentiment': sentiments
    })
    df.to_csv('sample_data/movie_reviews.csv', index=False)
    print("Generated movie_reviews.csv with", len(df), "rows.")

def generate_airline_passengers():
    np.random.seed(42)
    dates = pd.date_range(start='2010-01-01', end='2025-12-01', freq='MS')
    n = len(dates)
    
    # Passengers: upward trend + yearly seasonality + noise
    t = np.arange(n)
    trend = 100 + 1.8 * t
    seasonality = 40 * np.sin(2 * np.pi * t / 12) + 15 * np.cos(4 * np.pi * t / 12)
    noise = np.random.normal(0, 10, n)
    passengers = (trend + seasonality + noise).astype(int)
    passengers = np.clip(passengers, 50, None)
    
    # CargoTons: correlated with passengers (upward trend + seasonality + noise)
    cargo_trend = 50 + 0.9 * t
    cargo_seasonality = 15 * np.sin(2 * np.pi * t / 12 + 0.5)
    cargo_noise = np.random.normal(0, 5, n)
    cargo_tons = (cargo_trend + cargo_seasonality + cargo_noise).astype(int)
    cargo_tons = np.clip(cargo_tons, 20, None)
    
    # OilPrice: different trend and movement
    oil_trend = 60 + 0.1 * t
    oil_noise = np.random.normal(0, 8, n).cumsum()  # Random walk
    oil_price = np.round(oil_trend + oil_noise * 0.5, 2)
    oil_price = np.clip(oil_price, 35.0, 140.0)
    
    # Temperature: seasonal, no trend
    temp = np.round(15 + 12 * np.sin(2 * np.pi * (t - 4) / 12) + np.random.normal(0, 1.5, n), 1)
    
    df = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'Passengers': passengers,
        'CargoTons': cargo_tons,
        'OilPrice': oil_price,
        'Temperature': temp
    })
    df.to_csv('sample_data/airline_passengers.csv', index=False)
    print("Generated airline_passengers.csv with", len(df), "rows.")

if __name__ == '__main__':
    os.makedirs('sample_data', exist_ok=True)
    generate_iris()
    generate_house_prices()
    generate_movie_reviews()
    generate_airline_passengers()
