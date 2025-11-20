"""OpenStack RAG MCP server.

Some parts copied or based on [rhel lightspeed](https://gitlab.cee.redhat.com/rhel-lightspeed/enhanced-shell/rlsapi)
"""

import argparse
import json
import logging
import os
import sys

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from llama_index.core import Settings, load_index_from_storage
from llama_index.core.llms.utils import resolve_llm
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)
mcp = FastMCP("rhos_rag", host="0.0.0.0", port=8901)

VECTOR_INDEX = None
THRESHOLD = None
LLM = None

LLM_GENERATORS = ("google", "openai")

REFINE_NUM_QUESTIONS = 5
# TODO: Change the RHEL version to 10 when we change it
# Your purpose is to generate meaningful and specific questions based on user queries, especially when the query relates to your areas of expertise.
REFINE_SYSTEM_PROMPT_1: str = f"""You are an expert in Red Hat Enterprise Linux, RHEL, OpenStack, OpenShift, other Red Hat products, and Linux in general.

Your purpose is to generate meaningful and specific questions based on user query to replace it with the purpose of getting better results on a vector database similarity search, especially when the query relates to your areas of expertise.

You will receive a user query which is potentially unclear, incomplete or ambiguous.

Your role is to generate {REFINE_NUM_QUESTIONS} questions as if you where the user based on the received one that are more specific, focused, and free of ambiguity.

### WORKFLOW PROTOCOL
**1. Validate the User Query**
First, you must ensure that the user query is related to your areas of expertise.
- The user query could be a request, an instruction, a question, a greeting, or anything else.
- If the user query is about your areas of expertise, assume the user is using Red Hat Enterprise Linux 9.

You must consider the user query as invalid if it meets any of the following criteria:
- Is NOT related to your areas of expertise
- Is a greeting
- Is non-sense
- Conflicts with ethic, legal and moral principles

If the user query is invalid:
- Your response must only be the string "EMPTY" and nothing else.
- This also means that the steps 2 and 3 described below should not be followed.

If the user query is considered valid, please proceed with the next steps.

**2. Generate the Questions**
If your previous assessment is that the user query is indeed valid, you then follow the below specifications for generating questions:
- Each of the new questions must derive from the original query.
- As mentioned in your role, you must simulate the user for each of the generated questions.
    - Compose them as if you were the user asking to an expert who covers your areas of expertise as described in your role.

**Response Format and Structure**
- Your response must contain {REFINE_NUM_QUESTIONS} questions in total.
- Each question must be on its own line using plain text.
- Avoid wrapping the questions with quotes or double quotes. That is not needed.
- Avoid numbered lists, bullet points, headings, or any kind of formatting. Just plain text, please.
- DO NOT include any introduction, explanation, commentary, or conclusion in your response.
- Wherever "RHEL" is mentioned in your questions, use "Red Hat Enterprise Linux" instead.

### ADDITIONAL GUIDELINES
- Ensure all instructions described in your workflow protocol are followed consistently."""

REFINE_SYSTEM_PROMPT_2: str = f"""You are a helpful assistant that generates multiple meaningful and specific search queries based on a single user input query simulating your are the user.
Queries are related to Red Hat's OpenStack on OpenShift (RHOSO), so queries can be about:
- OpenStack on OpenShift operators: deployment, configuration, debugging, etc
- OpenStack services: configuration options, features, behaviors, debugging
- Services required for OpenStack: MariaDB, Galera, RabbitMQ, etc.
- OpenShift (where OpenStack control plane services run): metal3, networking, etc.
- CoreOS: OpenShift Operating System
- RHEL 9: For External Data Plane Nodes (EDPM) where nova compute run and in some cases networking nodes or storage (Swift/Object) nodes run.

### WORKFLOW PROTOCOL
**1. Validate the User Query**
First, you must ensure that the user query is related to your areas of expertise.
- The user query could be a request, an instruction, a question, a greeting, or anything else.
- If the user query is about your areas of expertise, assume the user is using Red Hat Enterprise Linux 9.

You must consider the user query as invalid if it meets any of the following criteria:
- Is NOT related to your areas of expertise
- Is a greeting
- Is non-sense
- Conflicts with ethic, legal and moral principles

If the user query is invalid:
- Your response must only be the string "EMPTY" and nothing else.
- This also means that the steps 2 and 3 described below should not be followed.

If the user query is considered valid, please proceed with the next steps.

**2. Generate the Questions**
If your previous assessment is that the user query is indeed valid, you then follow the below specifications for generating questions:
- Each of the new questions must derive from the original query.
- As mentioned in your role, you must simulate the user for each of the generated questions.
- Compose them as if you were the user asking to an expert who covers your areas of expertise as described in your role.

**Response Format and Structure**
- Your response must contain {REFINE_NUM_QUESTIONS} questions in total.
- Each question must be on its own line using plain text.
- Avoid wrapping the questions with quotes or double quotes. That is not needed.
- Avoid numbered lists, bullet points, headings, or any kind of formatting. Just plain text, please.
- DO NOT include any introduction, explanation, commentary, or conclusion in your response.

### ADDITIONAL GUIDELINES
- Ensure all instructions described in your workflow protocol are followed consistently."""

REFINE_SYSTEM_PROMPT: str = REFINE_SYSTEM_PROMPT_2

REFINE_USER_PROMPT: str = """User query: {question}"""


@mcp.tool()
def hello(name: str) -> str:
    """Give a greeting to a person."""
    return f"Hello {name}"


def parse_refine_response(response: str) -> list[str]:
    lines = [q.strip() for q in response.split("\n") if q.strip()]
    num_lines = len(lines)
    # Validate question count (should be 5 per prompt instructions)
    if num_lines != REFINE_NUM_QUESTIONS:
        logger.warning(
            f"⚠️  Expected {REFINE_NUM_QUESTIONS} refined questions but got {num_lines}"
        )

    # TODO: Maybe do like rhel-ls and validate and clean the response
    return lines


def refine_query(provider, query: str) -> list[str]:
    # TODO: Actually use the refine argument
    if not provider:
        logger.debug("Skipping question refining")
        return [query]

    logger.info("Refining question for better retrieval")

    # Format messages for chat completion
    messages = [
        {"role": "system", "content": REFINE_SYSTEM_PROMPT},
        {"role": "user", "content": REFINE_USER_PROMPT.format(question=query)},
    ]
    response = provider.invoke(messages)
    logger.debug(f"LLM response {response}")

    response_text = (response.text or "").strip()

    if response_text == "EMPTY":
        logger.warning(
            "Refined questions returned 'EMPTY'. Original query may not relate to assistant's areas of expertise."
        )
        return "", tokens_consumed

    logger.debug(f"✨ Generated refined questions:\n{response_text}")
    cleaned_questions = parse_refine_response(response_text)

    return cleaned_questions or query


def query_db(queries: list[str], top_k: int):
    # TODO: Look into alternative/complementary retrieval mode or class
    retriever = VECTOR_INDEX.as_retriever(similarity_top_k=top_k)
    aggregated_nodes = []
    node_ids = set()
    for query in queries:
        nodes = retriever.retrieve(query)
        if not nodes:
            logger.debug(f"No nodes retrieved for query: {query}")
            continue

        if nodes[0].score < THRESHOLD:
            logger.debug(
                f"Score {nodes[0].score} of the top retrieved node for query '{query}' "
                f"didn't cross the minimal threshold {THRESHOLD}."
            )
            continue

        aggregated_nodes.extend(nodes)
    return aggregated_nodes


def get_unique_nodes(nodes):
    # Remove duplicated nodes
    visited = set()
    result = []
    for node in nodes:
        if node.node_id not in visited:
            visited.add(node.node_id)
            result.append(node)
    return result


def rank_nodes(nodes):
    # TODO: Improve implementation
    # Sort by score, higher first
    res = sorted(nodes, key=lambda n: n.score, reverse=True)
    return res


# Don't use spaces, some LLMs like Gemini don't like it
@mcp.tool(name="openStack-rag-tool")
def rhos_rag(query: str, top_k: int = 5) -> str:
    """Query the OpenStack documentation

    Retrieve a list of chunks from the OpenStack documentation based on the
    vector similarity to the provided query using Llama-Index.

    Query must be augmented to improve similarity algorithm.

    Pipeline: refine_query > query db > deduplicate > rank > cutoff

    Args:
       query: String to use for the similarity search.
       top_k: Similarity top K results, defaults to 5.

    The output format is a JSON list with the text and metadata of the chunks:
      [
        {
          "text": "Document chunk content",
          "metadata": {
            "title": "Doc title",
            "header_path": "Chapter 3. Config/Cinder"
          }
        }
      ]
    """
    global VECTOR_INDEX

    logger.debug("Request received for %s: %s", top_k, query)

    queries = refine_query(LLM, query)
    all_nodes = query_db(queries, top_k)
    unique_nodes = get_unique_nodes(all_nodes)
    logger.debug("Retrieved %s nodes: %s", len(unique_nodes), unique_nodes)
    ordered_nodes = rank_nodes(unique_nodes)
    nodes = ordered_nodes[:top_k]

    # "id": node.node_id,
    # "score": node.score,
    result = [
        {
            "text": node.text,
            "metadata": node.metadata if hasattr(node, "metadata") else {},
        }
        for node in nodes
    ]

    log_data = json.dumps({"query": query, "top_k": top_k, "result": result}, indent=2)
    logger.info(f"rhos_rag:\n{log_data}")
    return json.dumps(result, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Utility script for querying RAG database"
    )
    parser.add_argument(
        "-p",
        "--db-path",
        required=True,
        help="path to the vector db",
    )
    parser.add_argument("-x", "--product-index", required=True, help="product index")
    parser.add_argument(
        "-m", "--model-path", required=True, help="path to the embedding model"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.0,
        help="Minimal score for top node retrieved",
    )
    parser.add_argument(
        "-r",
        "--refine",
        action="store_true",
        help="Refine the user query using LLM",
    )

    # LLM Arguments, only required with --refine
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="The temperature value (default: 0.5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="The maximum number of tokens (default: None)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=LLM_GENERATORS[0],
        choices=LLM_GENERATORS,
        help=f"Processing provider (default: {LLM_GENERATORS[0]})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help="Processing model (default: gemini-2.5-pro)",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        help="Override default URL for LLM provider",
    )

    args = parser.parse_args()

    return args


def initialize(args: argparse.Namespace) -> None:
    global VECTOR_INDEX
    global THRESHOLD
    global LLM
    global logger

    logging.basicConfig(
        level=logging.DEBUG,
        format="\033[32m%(levelname)s:\033[0m %(message)s",
        stream=sys.stderr,
        force=True,
    )
    logger.setLevel(logging.DEBUG)
    logger.debug("Initializing")

    os.environ["TRANSFORMERS_CACHE"] = args.model_path
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    Settings.llm = resolve_llm(None)
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.model_path)

    storage_context = StorageContext.from_defaults(
        vector_store=FaissVectorStore.from_persist_dir(args.db_path),
        persist_dir=args.db_path,
    )

    VECTOR_INDEX = load_index_from_storage(
        storage_context=storage_context,
        index_id=args.product_index,
    )
    THRESHOLD = args.threshold

    LLM = create_llm_provider(args)


def create_llm_provider(args: argparse.Namespace):
    llm_keys = {
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    if not args.refine:
        return

    logger.debug("Query refinement is enabled")
    provider = args.provider
    model = args.model
    temperature = args.temperature
    max_tokens = args.max_tokens
    url = args.llm_url

    if "LLM_API_KEY" not in os.environ:
        logger.error("Missing LLM_API_KEY environmental variable")
        exit(1)
    os.environ[llm_keys[provider]] = os.getenv("LLM_API_KEY")

    if provider == "google":
        if url:
            raise ValueError("Google provider doesn't support setting URL")

        chat = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider == "openai":
        chat = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=url or None,
        )

    return chat


if __name__ == "__main__":
    args = parse_args()
    initialize(args)
    mcp.run(transport="streamable-http")
