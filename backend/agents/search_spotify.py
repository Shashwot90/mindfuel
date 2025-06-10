import requests
from langchain_openai import ChatOpenAI
from google.colab import userdata
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import os 

class SpotifyAgent:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.rapidapi_key = os.getenv('RAPIDAPI_KEY')  # Replace with your RapidAPI key
        self.model = ChatOpenAI(api_key=self.openai_api_key, model="gpt-4o-mini")
        self.agent = create_react_agent(
            model=self.model,
            tools=[self.search_spotify_playlist],
            prompt="You are a helpful Spotify assistant. Your job is to help users discover Spotify playlists based on their search queries.",
            name="spotify_assistant"
        )

    @tool
    def search_spotify_playlist(self, query: str) -> str:
        """
        Search for a Spotify playlist by query using the RapidAPI Spotify Scraper API.
        Returns the top playlist result with its name, share URL, owner, and description.
        """
        url = "https://spotify-scraper.p.rapidapi.com/v1/search"
        params = {
            "term": query,
            "type": "playlist",
            "limit": "10"
        }
        headers = {
            "x-rapidapi-key": self.rapidapi_key,
            "x-rapidapi-host": "spotify-scraper.p.rapidapi.com"
        }
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            playlists = data.get("playlists", {}).get("items", [])
            if not playlists:
                return "No playlists found for your query."
            top_playlist = playlists[0]
            name = top_playlist.get("name", "Unknown Title")
            share_url = top_playlist.get("shareUrl", "No URL available")
            description = top_playlist.get("description", "No description available.")
            owner_info = top_playlist.get("owner", {})
            owner_name = owner_info.get("name", "Unknown owner")
            return (
                f"🎵 Playlist: {name}\n"
                f"👤 Created by: {owner_name}\n"
                f"📝 Description: {description}\n"
                f"🔗 Link: {share_url}"
            )
        except requests.exceptions.RequestException as e:
            return f"Error during Spotify search: {str(e)}"
        except (KeyError, IndexError, TypeError) as e:
            return f"Unexpected response format: {str(e)}"

    def run(self, user_input):
        response = self.agent.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
        return response["messages"][-1].content

# Example usage:
spotify_agent = SpotifyAgent()
result = spotify_agent.run("relaxing sleep music playlist")
print(result)