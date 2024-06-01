import connexion


def init_api():
    app = connexion.App(__name__)
    app.add_api("swagger.yaml")
    return app


def main():
    app = init_api()
    app.run(port=8084, debug=True)


if __name__ == "__main__":
    main()
