from graph_ql import graphql_app
from rest import app


# Add Graph QL Application to the FastAPI RESTFul Application
app.add_route("/graphql", graphql_app)
app.add_websocket_route("/graphql", graphql_app)
