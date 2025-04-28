import base64
import io
import time
from typing import List

from langchain.schema import Document
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from PIL import Image


class ImageEmbeddingModel:
    """
    A wrapper for different embedding models.
    """

    def __init__(self, model_name: str = "nomic-embed-text"):
        """
        Initialise the embedding model.

        Args:
            model_name: The name of the Ollama model to use for embeddings
        """
        self.embeddings_model = OllamaEmbeddings(model=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts."

        Args:
            texts: A list of texts to generate embeddings for

        Returns:
            A list of lists of embeddings
        """
        return self.embeddings_model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a single query text."

        Args:
            text: The text to generate embeddings for

        Returns:
            A list of embeddings
        """
        return self.embeddings_model.embed_query(text)


class ImageDescriptionGenerator:
    """
    Generate semantic descriptions of images using a multimodal model.
    """

    def __init__(self, model_name: str = "llava"):
        """
        Initialise the image description model.

        Args:
            model_name: The name of the Ollama model to use for generating descriptions
        """
        self.vision_model = ChatOllama(model=model_name)

    def encode_image(self, image_path: str) -> str:
        """
        Convert an image to a base64-encoded string.

        Args:
            image_path: The path to the image

        Returns:
            The base64-encoded string
        """
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize to model constraints - Hard-coded for now
            max_size = 768
            img.thumbnail((max_size, max_size))

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str

    def generate_descriptions(self, image_path: str) -> str:
        """
        Generate a semantic descriptions of an image.

        Args:
            image_path: The path to the image

        Returns:
            A semantic description of the image
        """
        img_base64 = self.encode_image(image_path)

        messages = [
            SystemMessage(
                content="You are an AI assistant that provides detailed semantic descriptions of images. Focus on the main subjects, their attributes, actions, relationships, the setting, colors, and any notable visual elements."
            ),
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": img_base64},
                    {
                        "type": "text",
                        "text": "Generate a detailed semantic description of this image. Focus on key visual elements that would be useful for semantic search.",
                    },
                ]
            ),
        ]

        response = self.vision_model.invoke(messages)
        return response.content


class ImageStore:
    """
    Store for images and their embeddings
    """

    def __init__(
        self,
        embedding_model: ImageEmbeddingModel,
        description_generator: ImageDescriptionGenerator,
        persist_directory: str = "image_store",
    ):
        """
        Initialise the image store.

        Args:
            embedding_model: The embedding model to use for generating embeddings
            description_generator: The description generator to use for generating descriptions
            persist_directory: The directory to persist the image store to
        """
        self.embedding_model = embedding_model
        self.description_generator = description_generator
        self.persist_directory = persist_directory

        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model.embeddings_model,
        )

    def add_image(self, image_path: str) -> str:
        """
        Add an image to the store.

        Args:
            image_path: The path to the image

        Returns:
            The id of the stored image
        """
        description = self.description_generator.generate_descriptions(image_path)
        doc = Document(page_content=description, metadata={"image_path": image_path})

        ids = self.vector_store.add_documents([doc])
        doc_id = ids[0]

        return doc_id

    def add_images(self, image_paths: List[str]) -> List[str]:
        """
        Add multiple images to the store.

        Args:
            image_paths: A list of paths to the images

        Returns:
            A list of ids of the stored images
        """
        descriptions = []
        docs = []

        total_images = len(image_paths)
        for idx, image_path in enumerate(image_paths):
            print(f'{idx+1}/{total_images} - Generating description for "{image_path}"')
            start_time = time.time()
            description = self.description_generator.generate_descriptions(image_path)
            duration = time.time() - start_time
            print(f"Description generated in {duration:.2f} seconds")
            doc = Document(
                page_content=description, metadata={"image_path": image_path}
            )
            descriptions.append(description)
            docs.append(doc)

        print(f"Adding {len(image_paths)} images to vector store...")
        ids = self.vector_store.add_documents(docs)

        print("Saving metadata...")

        return ids

    def get_image_count(self) -> int:
        """
        Get the number of images in the store.

        Returns:
            The number of images in the store
        """
        try:
            return len(self.vector_store.get()["documents"])
        except Exception as e:
            print(f"Error getting image count: {e}")
            return 0

    def search_by_image(self, image_path: str, k: int = 5) -> List[Document]:
        """
        Search for images similar to a given image.

        Args:
            image_path: The path to the query image
            k: The number of results to return

        Returns:
            List of Document objects containing similar images and their similarity scores
        """
        query_description = self.description_generator.generate_descriptions(image_path)
        results = self.vector_store.similarity_search_with_score(query_description, k=k)

        similar_images = []
        for doc, score in results:
            doc.metadata["similarity_score"] = score
            similar_images.append(doc)

        return similar_images


def create_image_store(
    embedding_model_name: str = "nomic-embed-text",
    vision_model_name: str = "llava",
    persist_directory: str = "image_store",
) -> ImageStore:
    """
    Create an image store.

    Args:
        embedding_model_name: The name of the embedding model to use
        description_model_name: The name of the description model to use
        persist_directory: The directory to persist the image store to

    Returns:
        Configured ImageStore instance
    """
    image_store = ImageStore(
        embedding_model=ImageEmbeddingModel(embedding_model_name),
        description_generator=ImageDescriptionGenerator(vision_model_name),
        persist_directory=persist_directory,
    )
    return image_store
