from pydantic import SecretStr


def test_no_auth_secret(mock_settings, mock_agent, test_client):
    """测试未设置 AUTH_SECRET 时允许所有请求。"""
    mock_settings.AUTH_SECRET = None
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer any-token"},
    )
    assert response.status_code == 200

    # 没有任何认证请求头时也应正常工作
    response = test_client.post("/invoke", json={"message": "test"})
    assert response.status_code == 200


def test_auth_secret_correct(mock_settings, mock_agent, test_client):
    """测试设置 AUTH_SECRET 后允许携带正确令牌的请求。"""
    mock_settings.AUTH_SECRET = SecretStr("test-secret")
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer test-secret"},
    )
    assert response.status_code == 200


def test_auth_secret_incorrect(mock_settings, mock_agent, test_client):
    """测试设置 AUTH_SECRET 后拒绝携带错误令牌的请求。"""
    mock_settings.AUTH_SECRET = SecretStr("test-secret")
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer wrong-secret"},
    )
    assert response.status_code == 401

    # 也应拒绝不带认证请求头的请求
    response = test_client.post("/invoke", json={"message": "test"})
    assert response.status_code == 401
