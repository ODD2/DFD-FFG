import os
import logging
import requests


def send_to_telegram(message):
    apiToken = os.getenv("API_TOKEN")
    chatID = os.getenv("CHAT_ID")
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'
    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
    except Exception as e:
        logging.error(f"Telegram Notification Failed: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='message', type=str)
    args = parser.parse_args()
    send_to_telegram(args.message)
