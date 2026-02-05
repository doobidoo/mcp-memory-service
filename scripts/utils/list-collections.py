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


def list_collections():
    try:
        # Connect to local ChromaDB
        client = HttpClient(host="localhost", port=8000)

        # List all collections
        collections = client.list_collections()

        print("\nFound Collections:")
        print("------------------")
        for collection in collections:
            print(f"Name: {collection.name}")
            print(f"Metadata: {collection.metadata}")
            print(f"Count: {collection.count()}")
            print("------------------")

    except Exception as e:
        print(f"Error connecting to local ChromaDB: {str(e)}")


if __name__ == "__main__":
    list_collections()
