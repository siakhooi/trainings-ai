# import libraries
import weaviate
import cohere

# Add your Cohere API key here
# You can obtain a key by signing up in https://dashboard.cohere.com/ or https://docs.cohere.com/reference/key
cohere_api_key = ''

# Create a Cohere client object
co = cohere.Client(cohere_api_key)

# Create a vector database object
# Connect to the Weaviate demo databse containing 10M wikipedia vectors
# You can obtain a key from https://weaviate.io and refer to https://weaviate.io/developers/wcs/authentication#connect-with-an-api

auth_config = weaviate.auth.AuthApiKey(api_key="")
client = weaviate.Client(
    url="https://cohere-demo.weaviate.network/",
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": cohere_api_key,
    }
)

# Dense Retrieval function
def dense_retrieval(query, results_lang='en', num_results=3):

    nearText = {"concepts": [query]}
    properties = ["text", "title", "url", "_additional {distance}"]

    # To filter by language
    where_filter = {
        "path": ["lang"],
        "operator": "Equal",
        "valueString": results_lang
        }
    response = (
        client.query
        .get("Articles", properties)
        .with_near_text(nearText)
        .with_where(where_filter)
        .with_limit(num_results)
        .do()
    )

    result = response['data']['Get']['Articles']
    return result

query = "What is the highest mountain peak in the world?"
top_documents = dense_retrieval(query)

top_documents

query1 = "What is the capital of United States of America?"
# Example usage
top_documents1 = dense_retrieval(query1)

top_documents1