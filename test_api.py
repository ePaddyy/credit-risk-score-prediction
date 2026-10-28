
import pytest

from predict import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def valid_applicant():
    return {
        "seniority": 3,
        "home": "owner",
        "time": 36,
        "age": 69,
        "marital": "married",
        "records": "no",
        "job": "fixed",
        "expenses": 75,
        "income": 120.0,
        "assets": 3000.0,
        "debt": 0.0,
        "amount": 1000000,
        "price": 1000,
    }


def test_ping(client):
    response = client.get("/ping")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_valid_prediction(client):
    response = client.post("/predict", json=valid_applicant())

    assert response.status_code == 200

    result = response.get_json()
    assert 0 <= result["default_probability"] <= 1
    assert isinstance(result["default"], bool)


def test_missing_field(client):
    applicant = valid_applicant()
    applicant.pop("income")

    response = client.post("/predict", json=applicant)

    assert response.status_code == 400
    assert "Missing required fields" in response.get_json()["error"]


def test_invalid_category(client):
    applicant = valid_applicant()
    applicant["home"] = "unknown_home"

    response = client.post("/predict", json=applicant)

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid input"


def test_invalid_numeric_field(client):
    applicant = valid_applicant()
    applicant["age"] = "sixty-nine"

    response = client.post("/predict", json=applicant)

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid input"


def test_null_numeric_field(client):
    applicant = valid_applicant()
    applicant["income"] = None

    response = client.post("/predict", json=applicant)

    assert response.status_code == 200


def test_non_json_request(client):
    response = client.post(
        "/predict",
        data="not json",
        content_type="text/plain",
    )

    assert response.status_code == 415
