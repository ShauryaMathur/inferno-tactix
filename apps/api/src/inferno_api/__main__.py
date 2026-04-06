from .app import app, API_HOST, API_PORT


if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT, debug=True)
