# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from chromadb import HttpClient


def test_connection(port=8000):
    try:
        # Try to connect to local ChromaDB
        client = HttpClient(host="localhost", port=port)
        # Try a simple operation
        heartbeat = client.heartbeat()
        print(f"Successfully connected to ChromaDB on port {port}")
        print(f"Heartbeat: {heartbeat}")

        # List collections
        collections = client.list_collections()
        print("\nFound collections:")
        for collection in collections:
            print(f"- {collection.name} (count: {collection.count()})")

    except Exception as e:
        print(f"Error connecting to ChromaDB on port {port}: {str(e)}")


if __name__ == "__main__":
    # Try default port
    test_connection()

    # If the above fails, you might want to try other common ports:
    # test_connection(8080)
    # test_connection(9000)
