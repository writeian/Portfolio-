import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inject custom CSS to adjust column widths in the table
st.markdown(
    """
    <style>
    /* Target the second column (course id) */
    table[data-testid="stTableStyledTable"] th:nth-child(2),
    table[data-testid="stTableStyledTable"] td:nth-child(2) {
        width: 8ch !important;
        max-width: 8ch !important;
        white-space: normal !important;
    }
    /* Target the third column (course title) */
    table[data-testid="stTableStyledTable"] th:nth-child(3),
    table[data-testid="stTableStyledTable"] td:nth-child(3) {
        width: 18ch !important;
        max-width: 18ch !important;
        white-space: normal !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    file_path = r"C:\Users\HP\Documents\Data for AI course selection project .csv"
    df = pd.read_csv(file_path, encoding='cp1252')
    # Standardize column names: strip whitespace and convert to lowercase
    df.columns = df.columns.str.strip().str.lower()
    # Ensure text columns have string values, fill missing values, and convert to lowercase
    df['course title'] = df['course title'].fillna('').astype(str).str.lower().str.strip()
    df['course description'] = df['course description'].fillna('').astype(str).str.lower().str.strip()
    df['keywords'] = df['keywords'].fillna('').astype(str).str.lower().str.strip()
    # Combine text fields for vectorization
    df['combined'] = df['course title'] + " " + df['course description'] + " " + df['keywords']
    return df

df = load_data()

# Create TF-IDF vectors from the 'combined' text field
vectorizer = TfidfVectorizer(stop_words='english')
course_vectors = vectorizer.fit_transform(df['combined'])
cosine_sim = cosine_similarity(course_vectors, course_vectors)

def get_recommendations(course_title, cosine_sim=cosine_sim, df=df, top_n=5):
    # Map course titles to their DataFrame indices
    indices = pd.Series(df.index, index=df['course title']).drop_duplicates()
    try:
        idx = indices[course_title]
    except KeyError:
        st.error("Course not found. Please check the title.")
        return pd.DataFrame()
    
    # Compute similarity scores and sort them (excluding the input course)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    course_indices = [i[0] for i in sim_scores]
    
    # Retrieve the recommended courses
    result_df = df.iloc[course_indices][['course id', 'course title', 'course description']].copy()
    
    # Capitalize the course title (title-case)
    result_df['course title'] = result_df['course title'].str.title()
    
    # Capitalize the course description so that each sentence starts with an uppercase letter.
    def sentence_case(text):
        sentences = text.split('. ')
        return '. '.join(s.capitalize() for s in sentences)
    
    result_df['course description'] = result_df['course description'].apply(sentence_case)
    
    return result_df

st.title("SmartCourse Navigator")
st.write(
    "This tool uses advanced AI to analyze course content and generate personalized recommendations. "
    "Select a course from the dropdown menu to see a curated list of similar courses. "
    "The results table shows each course's ID, title, and a formatted description for easy reading."
)


# Use a dropdown menu for course selection
course_input = st.selectbox("Choose a course", df['course title'].unique())

if course_input:
    recommendations = get_recommendations(course_input)
    # Capitalize the output column headings
    recommendations.columns = recommendations.columns.str.title()
    st.subheader("Recommended Courses:")
    st.table(recommendations)
