import requests


def scrape_linkdin_profile(linkdin_profile_url: str, mock: bool = False):
    if mock:
        linkdin_profile_url = "https://gist.githubusercontent.com/emarco177/859ec7d786b45d8e3e3f688c6c9139d8/raw/5eaf8e46dc29a98612c8fe0c774123a7a2ac4575/eden-marco-scrapin.json"

        response = requests.get(
            linkdin_profile_url,
            timeout=10,
        )

        return response.json()


if __name__ == "__main__":
    print(scrape_linkdin_profile("", mock=True))
