import requests
from langchain_openai import ChatOpenAI
from google.colab import userdata
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import os

class YoutubeAgent:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.rapidapi_key = os.getenv('RAPIDAPI_KEY') # rapidapi_key or "api key here"  # Replace with your RapidAPI key
        self.model = ChatOpenAI(api_key=self.openai_api_key, model="gpt-4o-mini")
        self.agent = create_react_agent(
            model=self.model,
            tools=[self.search_youtube_video],
            prompt="You are a helpful YouTube video assistant. Your job is to help users find YouTube videos based on their search queries.",
            name="youtube_assistant"
        )

    @tool
    def search_youtube_video(self, query: str, region: str = "US", language: str = "en") -> str:
        """
        Search for a video on YouTube using the RapidAPI YouTube138 API.
        Args:
            query (str): The search query (e.g., video title or keyword).
            region (str): The region code (default is 'US').
            language (str): The language code (default is 'en').
        Returns:
            str: The title and URL of the top video result.
        """
        url = "https://youtube138.p.rapidapi.com/search/"
        params = {
            "q": query,
            "hl": language,
            "gl": region
        }
        headers = {
            "x-rapidapi-key": self.rapidapi_key,
            "x-rapidapi-host": "youtube138.p.rapidapi.com"
        }
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            for result in data.get("contents", []):
                video = result.get("video")
                if video:
                    title = video.get("title")
                    video_id = video.get("videoId")
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    return f"Top result:\nTitle: {title}\nURL: {video_url}"
            return "No video found for your query."
        except requests.exceptions.RequestException as e:
            return f"Error during YouTube search: {str(e)}"
        except (KeyError, IndexError, TypeError) as e:
            return f"Unexpected response format: {str(e)}"

    def run(self, user_input):
        response = self.agent.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
        return response["messages"][-1].content
    
    
# Usage example for YoutubeAgent class
youtube_agent = YoutubeAgent()
result = youtube_agent.run("10 minute guided meditation video")
print(result)