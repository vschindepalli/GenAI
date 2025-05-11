import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError
import os
import json
import logging
import time
import requests
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, filename="app.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Grok API client
client = OpenAI(
    api_key="your_api_key",
    base_url="https://api.x.ai/v1"
)

# Function to check server connectivity
def check_server_status():
    try:
        response = requests.get("https://api.x.ai/v1", timeout=5)
        if response.status_code in [404, 401]:
            return {"status": "connected", "message": "API server is reachable"}
        return {"status": response.status_code, "message": response.text[:100]}
    except requests.RequestException as e:
        return {"status": None, "message": f"Network error: {str(e)}"}

# Function to fetch and analyze X posts (disable caching)
@st.cache_data(show_spinner=False, ttl=60)  # Cache for 60 seconds, but refresh on max_posts change
def analyze_trends(keyword, max_posts=50, retries=2):
    for attempt in range(retries):
        try:
            # Validate input
            if not keyword.strip():
                return {"error": "Keyword cannot be empty."}
            
            # Construct prompt for Grok API
            prompt = f"""
            Analyze exactly {max_posts} recent public posts on X about '{keyword}'.
            Provide a JSON object with:
            1. "summary": A summary of the main topics discussed (max 100 words, string).
            2. "sentiment": Percentages for positive, negative, and neutral posts (e.g., {{"positive": 40, "negative": 20, "neutral": 40}}).
            3. "posts": A list of up to 5 example post texts (strings, no user info).
            Ensure the sentiment percentages are based on the {max_posts} posts analyzed.
            """
            logger.info(f"API prompt: {prompt}")

            # Call Grok API with unique timestamp to avoid caching
            response = client.chat.completions.create(
                model="grok-3-mini-beta",
                messages=[
                    {"role": "system", "content": "You are a social media trend analyzer."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                extra_headers={"X-Timestamp": str(datetime.now())}  # Prevent API caching
            )

            # Parse response
            result = json.loads(response.choices[0].message.content)
            logger.info(f"API response: {result}")
            logger.info(f"Raw sentiment data: {result.get('sentiment', 'No sentiment data')}")
            return result

        except AuthenticationError as e:
            logger.error(f"Authentication error: {str(e)}")
            return {"error": "Invalid API key. Please check your GROK_API_KEY environment variable."}
        except RateLimitError as e:
            logger.error(f"Rate limit error: {str(e)}")
            return {"error": "API rate limit exceeded. Please wait and try again later."}
        except APIConnectionError as e:
            logger.error(f"Connection error: {str(e)}")
            if attempt < retries - 1:
                logger.info(f"Retrying... Attempt {attempt + 2}/{retries}")
                time.sleep(2 ** attempt)
                continue
            return {"error": "Failed to connect to the Grok API. Check your network or API status at https://status.x.ai."}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            if "resource not found" in str(e).lower():
                return {"error": "Resource not found. Check the model name or contact support@x.ai."}
            if attempt < retries - 1:
                logger.info(f"Retrying... Attempt {attempt + 2}/{retries}")
                time.sleep(2 ** attempt)
                continue
            return {"error": f"Failed to analyze trends: {str(e)}"}

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header { font-size: 2.5em; color: #1E90FF; font-weight: bold; }
    .sub-header { font-size: 1.5em; color: #4682B4; }
    .summary-box {
        background-color: #F0F8FF;
        color: #333333;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #4682B4;
        font-size: 1em;
    }
    .stTable { font-size: 0.9em; }
    </style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.markdown('<div class="main-header">Real-Time Social Media Trend Analyzer</div>', unsafe_allow_html=True)
st.write("Analyze trending topics or hashtags on X with AI-powered insights.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    keyword = st.text_input("Keyword/Hashtag", value="#AI", help="Enter a keyword or hashtag to analyze (e.g., #AI, Tesla).")
    max_posts = st.slider("Max Posts to Analyze", 10, 100, 50, help="Maximum number of posts to fetch.")
    st.markdown("---")
    st.header("About")
    st.write("This app uses xAI's Grok API to analyze X posts in real-time. Enter a keyword or hashtag to see trends, sentiment, and example posts.")
    st.markdown("[Learn more about Grok API](https://x.ai/api)")
    # Display server status
    st.header("API Status")
    status = check_server_status()
    if status["status"] == "connected":
        st.success(f"Server Status: Connected ({status['message']})")
    elif status["status"]:
        st.write(f"Server Status: {status['status']} ({status['message'][:50]}...)")
    else:
        st.warning(f"Server Unreachable: {status['message']}")

# Main content
if st.button("Analyze Trends", key="analyze_button"):
    if not keyword.strip():
        st.error("Please enter a valid keyword or hashtag.")
    else:
        with st.spinner("Fetching and analyzing X posts..."):
            result = analyze_trends(keyword, max_posts)

            if "error" in result:
                st.error(result["error"])
                if "rate limit" in result["error"].lower():
                    st.info("Check your API quota at https://x.ai/api or reduce the number of posts.")
                elif "connection" in result["error"].lower():
                    st.info("Try checking your internet connection or the API status at https://status.x.ai.")
                elif "resource not found" in result["error"].lower():
                    st.info("Ensure the model (grok-2) is available. Contact support@x.ai or check https://docs.x.ai/.")
            else:
                # Summary
                st.markdown('<div class="sub-header">Summary</div>', unsafe_allow_html=True)
                summary = result.get("summary", "No summary available.")
                st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

                # Sentiment Analysis
                st.markdown('<div class="sub-header">Sentiment Analysis</div>', unsafe_allow_html=True)
                sentiment = result.get("sentiment", {"positive": 0, "negative": 0, "neutral": 0})
                # Convert sentiment values to floats to handle strings or floats
                try:
                    sentiment_values = {
                        "positive": float(sentiment.get("positive", 0)),
                        "negative": float(sentiment.get("negative", 0)),
                        "neutral": float(sentiment.get("neutral", 0))
                    }
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid sentiment data format: {sentiment}, error: {str(e)}")
                    sentiment_values = {"positive": 0, "negative": 0, "neutral": 0}
                logger.info(f"Processed sentiment: {sentiment_values}")
                
                df_sentiment = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative", "Neutral"],
                    "Percentage": [
                        sentiment_values["positive"],
                        sentiment_values["negative"],
                        sentiment_values["neutral"]
                    ]
                })
                # Display sentiment percentages for debugging
                st.write(f"Sentiment for {max_posts} posts: Positive {sentiment_values['positive']}%, Negative {sentiment_values['negative']}%, Neutral {sentiment_values['neutral']}%")
                
                # Check if sentiment data is valid (non-zero and numeric)
                if df_sentiment["Percentage"].sum() == 0 or not all(isinstance(x, (int, float)) for x in df_sentiment["Percentage"]):
                    st.warning("No valid sentiment data available. Try a different keyword or more posts.")
                    # Fallback chart with placeholder data
                    df_sentiment = pd.DataFrame({
                        "Sentiment": ["Positive", "Negative", "Neutral"],
                        "Percentage": [33.3, 33.3, 33.4]
                    })
                    fig = px.pie(
                        df_sentiment,
                        names="Sentiment",
                        values="Percentage",
                        title="Sentiment Distribution (Placeholder)",
                        color_discrete_sequence=["#00FF7F", "#FF4500", "#1E90FF"]
                    )
                else:
                    fig = px.pie(
                        df_sentiment,
                        names="Sentiment",
                        values="Percentage",
                        title=f"Sentiment Distribution ({max_posts} Posts)",
                        color_discrete_sequence=["#00FF7F", "#FF4500", "#1E90FF"]
                    )
                fig.update_traces(textinfo="percent+label", hoverinfo="label+percent+value")
                st.plotly_chart(fig, use_container_width=True)

                # Example Posts
                st.markdown('<div class="sub-header">Example Posts</div>', unsafe_allow_html=True)
                posts = result.get("posts", [])
                if posts:
                    df_posts = pd.DataFrame(posts, columns=["Post"])
                    st.table(df_posts.style.set_properties(**{'text-align': 'left'}))
                else:
                    st.warning("No example posts available.")

# Footer
st.markdown("---")
st.markdown("Powered by xAI's Grok API | Built with Streamlit | © 2025")