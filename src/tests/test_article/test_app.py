import json
import pathlib

import pytest
from jsonschema import RefResolver, validate

from blog.app import app
from blog.models import Article


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def validate_payload(payload, schema_name):
    """
    Validate payload with selected schema
    """
    schemas_dir = str(f"{pathlib.Path(__file__).parent.absolute()}/schemas")
    schema = json.loads(pathlib.Path(f"{schemas_dir}/{schema_name}").read_text())
    validate(
        payload,
        schema,
        resolver=RefResolver(
            "file://" + str(pathlib.Path(f"{schemas_dir}/{schema_name}").absolute()),
            schema,  # it's used to resolve the file inside schemas correctly
        ),
    )


def test_create_article(client):
    """
    GIVEN request data for new article
    WHEN endpoint /create-article/ is called
    THEN it should return Article in json format that matches the schema
    """
    data = {
        "author": "john@doe.com",
        "title": "New Article",
        "content": "Some extra awesome content",
    }
    response = client.post(
        "/create-article/",
        data=json.dumps(data),
        content_type="application/json",
    )

    validate_payload(response.json, "article.json")


def test_get_article(client):
    """
    GIVEN ID of article stored in the database
    WHEN endpoint /article/<id-of-article>/ is called
    THEN it should return Article in json format that matches the schema
    """
    article = Article(
        author="jane@doe.com",
        title="New Article",
        content="Super extra awesome article",
    ).save()
    response = client.get(
        f"/article/{article.id}/",
        content_type="application/json",
    )

    validate_payload(response.json, "article.json")


def test_list_articles(client):
    """
    GIVEN articles stored in the database
    WHEN endpoint /article-list/ is called
    THEN it should return list of Article in json format that matches the schema
    """
    Article(
        author="jane@doe.com",
        title="New Article",
        content="Super extra awesome article",
    ).save()
    response = client.get(
        "/article-list/",
        content_type="application/json",
    )

    validate_payload(response.json, "articlelist.json")


@pytest.mark.parametrize(
    "data",
    [
        {
            "author": "John Doe",
            "title": "New Article",
            "content": "Some extra awesome content",
        },
        {
            "author": "John Doe",
            "title": "New Article",
        },
        {"author": "John Doe", "title": None, "content": "Some extra awesome content"},
    ],
)
def test_create_article_bad_request(client, data):
    """
    GIVEN request data with invalid values or missing attributes
    WHEN endpoint /create-article/ is called
    THEN it should return status 400
    """
    response = client.post(
        "/create-article/",
        data=json.dumps(data),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json is not None
