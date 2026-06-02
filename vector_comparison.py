from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_vector_distance(original_text, final_text):
    """
    כלי למדידת מרחק וקטורי בין המשפט המקורי למשפט הסופי לאחר שרשרת התרגומים.
    """
    # יצירת המודל שיהפוך את הטקסטים לווקטורים
    vectorizer = TfidfVectorizer()
    
    # המרת שני המשפטים למטריצה של ווקטורים (Embedding)
    try:
        tfidf_matrix = vectorizer.fit_transform([original_text, final_text])
    except ValueError:
        return "Error: Texts are empty or invalid."
    
    # חישוב הדמיון (Cosine Similarity) - 1.0 אומר זהה לחלוטין, 0.0 אומר שונה לחלוטין
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # חישוב המרחק הווקטורי (מרחק = 1 פחות הדמיון)
    distance = 1 - similarity
    
    return {
        "original_text": original_text,
        "final_text": final_text,
        "similarity_score": round(similarity, 4),
        "vector_distance": round(distance, 4) # מרחק 0 = חזרה מדויקת של המשפט
    }

# בדיקה פנימית של הכלי (טסט של האורקסטרטור)
if __name__ == "__main__":
    original = "One for all and all for one"
    final = "All for one and one for all"
    
    result = calculate_vector_distance(original, final)
    print("--- Vector Comparison Tool Results ---")
    print(f"Original: {result['original_text']}")
    print(f"Final:    {result['final_text']}")
    print(f"Similarity Score: {result['similarity_score']}")
    print(f"Vector Distance:  {result['vector_distance']}")
