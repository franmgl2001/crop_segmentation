from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# Your client credentials
client_id = "d56cd375-6541-489f-8c01-a83ca576fe79"
client_secret = "a83lrnFz6OsXRyNAbwpZd0t2FxC3yd0U"


# Token URL for Sentinel Hub
token_url = (
    "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
)


def fetch_new_token():
    """Fetches a new access token using the client credentials grant."""
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)

    # Fetch token for the session
    token = oauth.fetch_token(
        token_url=token_url,
        client_id=client_id,
        client_secret=client_secret,
        include_client_id=True,
    )
    return token


# Fetch initial token
token = fetch_new_token()
oauth = OAuth2Session(client_id=client_id, token=token)


def make_request_with_retry():
    """Makes a request and handles token expiration by retrying."""
    try:
        # Make the API request
        response = oauth.get(
            "https://services.sentinel-hub.com/configuration/v1/wms/instances"
        )

        # Check the response status
        if response.status_code == 200:
            print("Request was successful.")
            print(response.json())  # Print the JSON response content
        elif response.status_code == 401:  # Token expired, needs refresh
            print("Access token expired, fetching a new one...")
            token = fetch_new_token()  # Fetch a new token
            oauth.token = token  # Update the session token
            response = oauth.get(
                "https://services.sentinel-hub.com/configuration/v1/wms/instances"
            )  # Retry the request
            if response.status_code == 200:
                print("Request was successful after refreshing the token.")
                print(response.json())
            else:
                print(
                    f"Request failed after retry: {response.status_code} - {response.text}"
                )
        else:
            print(f"Request failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Make the request and handle token expiration
make_request_with_retry()
