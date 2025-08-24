# MCP Memory Service Invocation Guide

This document outlines the specific keywords and phrases that trigger the MCP memory service in Claude Desktop conversations, along with examples of how to use them effectively.

## Memory Storage Keywords

The following keywords and phrases will trigger Claude to store information in the MCP memory service:

| Keyword/Phrase | Example Usage | Notes |
|----------------|---------------|-------|
| "remember that" | "Please remember that my project deadline is May 15th." | Most common and reliable trigger |
| "remember this" | "Remember this: The database credentials are in the config file." | Works well for specific information |
| "save to memory" | "Save to memory that I prefer dark mode in all applications." | Explicit storage command |
| "store in memory" | "Store in memory that my team meeting is every Tuesday." | Explicit storage command |
| "add to memory" | "Add to memory that the client prefers email communication." | Works for appending information |
| "make a note" | "Make a note that we need to follow up next week." | More conversational trigger |
| "remember for later" | "Remember for later that we discussed the API integration approach." | Implies future retrieval |
| "keep track of" | "Keep track of the fact that version 2.3 has this bug." | Good for tracking issues |

## Memory Retrieval Keywords

The following keywords and phrases will trigger Claude to retrieve information from the MCP memory service:

| Keyword/Phrase | Example Usage | Notes |
|----------------|---------------|-------|
| "do you remember" | "Do you remember my favorite color?" | Natural question format |
| "recall" | "Recall what we discussed about the database schema." | Direct command for retrieval |
| "what did I tell you about" | "What did I tell you about my project timeline?" | Conversational retrieval |
| "retrieve from memory" | "Retrieve from memory our discussion about authentication." | Explicit retrieval command |
| "search your memory for" | "Search your memory for information about the API design." | Useful for specific searches |
| "find in your memory" | "Find in your memory our discussion from last week." | Good for time-based retrieval |
| "check your memory for" | "Check your memory for the requirements we outlined." | Alternative retrieval phrase |
| "what do you know about" | "What do you know about my team structure?" | Natural question format |

### How Retrieved Information Works in Conversation

When Claude retrieves information from the MCP memory service, it's important to understand how this affects the current conversation:

1. **Automatic Context Integration**: When memories are retrieved, they are automatically included in Claude's context for the current conversation. This means Claude will immediately incorporate and reference this information in its responses.

2. **Trial and Error Retrieval**: If your first retrieval attempt doesn't find the information you're looking for, Claude's responses about "no matching memories" do NOT get stored in the memory service. You can safely make multiple retrieval attempts without cluttering the memory database.

3. **Selective Memory Storage**: Just because information was retrieved doesn't mean the current conversation about that retrieval is automatically stored. The current conversation will only be stored if you explicitly ask Claude to remember it using one of the memory storage keywords.

### Multi-Step Retrieval Workflow

A common workflow for effectively using the memory service:

1. **Initial Retrieval**: "Search your memory for information about the marketing campaign we discussed."

2. **Refinement**: If the first search isn't what you need, you can refine it: "Try searching for memories with the tag 'Q3 marketing' instead."

3. **Further Refinement**: "Check if there's anything from conversations we had in March about social media strategy."

4. **Found Relevant Information**: Once Claude retrieves the information you want, you can continue your conversation based on this information.

5. **Explicit Storage**: If you want the current conversation to be remembered: "Please remember this conversation about our refined marketing approach with tags 'marketing strategy' and 'revised'."

This approach lets you search through past memories with multiple attempts until you find what you need, without storing all the search attempts, and then explicitly store only the valuable parts of the current conversation.

### Behind-the-Scenes Example: Retrieving Mistral Model Information

Let's walk through a complete example showing what happens behind the scenes when retrieving and using memory:

**Current Conversation Context**: Human and Claude are discussing coreference resolution using FastCoref and Ollama Mistral.

#### Step 1: Initial Retrieval Attempt

**Human**: "Do you remember which Mistral model I said I was planning to use?"

**Behind the Scenes**:
- Claude recognizes the phrase "Do you remember" as a memory retrieval trigger
- Claude calls the MCP memory service using the `retrieve_memory` function:
  ```
  retrieve_memory({"query": "Mistral model user plans to use", "n_results": 5})
  ```
- MCP searches the ChromaDB vector database using the query embedding
- MCP finds a few memories about Mistral but not the specific model information
- MCP returns these results to Claude

**Claude**: "I found some information about Mistral in my memory, but I don't see specifics about which model you were planning to use. Would you like me to share what I found about Mistral in general?"

#### Step 2: Refined Retrieval Attempt

**Human**: "Try checking memories with the tag 'models' or anything from our conversation last Tuesday."

**Behind the Scenes**:
- Claude recognizes this as a refined memory retrieval request
- Claude makes two MCP function calls:
  ```
  search_by_tag({"tags": ["models"]})
  ```
  and
  ```
  recall_memory({"query": "conversation from last Tuesday about Mistral", "n_results": 5})
  ```
- MCP processes both requests, checking for tag matches and time-based retrieval
- MCP still doesn't find the exact information and returns partial matches

**Claude**: "I found some memories tagged 'models' but nothing specific about which Mistral model you plan to use. I did find a conversation from last Tuesday, but it was about a different topic."

#### Step 3: More Specific Retrieval

**Human**: "Search your memory for anything I mentioned about 'Mistral-7B' specifically."

**Behind the Scenes**:
- Claude calls MCP again with a more specific query:
  ```
  retrieve_memory({"query": "Mistral-7B model mentioned by user", "n_results": 5})
  ```
- MCP searches with this specific term
- MCP finds a relevant memory about the user planning to use Mistral-7B-Instruct
- MCP returns this memory to Claude

**Claude**: "I found it! In a previous conversation, you mentioned that you planned to use the Mistral-7B-Instruct model with Ollama because it has good performance for coreference resolution while being efficient enough to run on your local hardware."

#### Step 4: Continuing the Productive Conversation

**Human**: "Great! So based on that, how should I configure FastCoref to work optimally with Mistral-7B-Instruct via Ollama?"

**Behind the Scenes**:
- Claude has now incorporated the retrieved memory into its current conversation context
- No additional MCP calls are made at this point
- Claude uses both the retrieved information and its general knowledge to formulate a response

**Claude**: [Provides detailed information about FastCoref configuration with Mistral-7B-Instruct]

#### Step 5: Explicitly Saving the Valuable Conversation

**Human**: "This was very helpful. Please remember this conversation about FastCoref configuration with Mistral-7B-Instruct and tag it as 'coreference', 'configuration', and 'mistral'."

**Behind the Scenes**:
- Claude recognizes "Please remember" as a memory storage trigger
- Claude calls the MCP memory storage function:
  ```
  store_memory({
    "content": "Detailed conversation about configuring FastCoref with Mistral-7B-Instruct via Ollama, including settings for optimal coreference resolution.",
    "metadata": {
      "tags": "coreference,configuration,mistral",
      "type": "technical_discussion"
    }
  })
  ```
- MCP stores this information in ChromaDB with proper embeddings and metadata
- MCP confirms successful storage to Claude

**Claude**: "I've saved our conversation about FastCoref configuration with Mistral-7B-Instruct and tagged it as requested. You can retrieve this information in future conversations."

This example illustrates how multiple retrieval attempts can be made without cluttering memory, and how only the valuable parts of the conversation are stored when explicitly requested. The MCP service handles the actual storage and retrieval while Claude manages the conversation flow and context integration.

## Tag-Based Retrieval

To retrieve memories based on specific tags, use these phrases:

| Keyword/Phrase | Example Usage | Notes |
|----------------|---------------|-------|
| "find memories with tag" | "Find memories with tag 'project'." | Explicit tag retrieval |
| "search for tag" | "Search for tag 'meeting notes'." | Simple tag search |
| "retrieve memories tagged" | "Retrieve memories tagged 'important'." | Alternative tag phrase |
| "recall information tagged" | "Recall information tagged 'deadline'." | Combines recall with tags |

## Time-Based Retrieval

For retrieving memories based on when they were stored, use these time expressions:

| Time Expression | Example Usage | Notes |
|----------------|---------------|-------|
| "yesterday" | "What did we discuss yesterday?" | Simple day reference |
| "last week" | "Recall our conversation from last week." | Weekly timeframe |
| "two days ago" | "What did I tell you two days ago?" | Specific day count |
| "this morning" | "What did we talk about this morning?" | Time of day reference |
| "last month" | "Find our discussions from last month about the project." | Monthly timeframe |
| "earlier today" | "What did I ask you to remember earlier today?" | Same-day reference |

## Memory Deletion Keywords

To delete specific memories or all memories with certain tags:

| Keyword/Phrase | Example Usage | Notes |
|----------------|---------------|-------|
| "forget" | "Please forget what I told you about my address." | Used for specific deletion |
| "delete from memory" | "Delete from memory our discussion about the old project." | Explicit deletion |
| "remove from memory" | "Remove from memory everything with tag 'temporary'." | Good for tag-based cleanup |
| "clear memories about" | "Clear memories about the prototype discussion." | Content-based deletion |

## Best Practices

1. **Be explicit**: When storing important information, use clear commands like "Please remember that..." rather than assuming Claude will store it automatically.

2. **Use tags when storing**: Add tags to organize memories: "Remember that my favorite color is blue, tag this as 'preferences'."

3. **Be specific when retrieving**: The more specific your retrieval request, the more relevant the results will be.

4. **Combine approaches**: For complex retrievals, combine techniques: "Find memories from last week tagged 'project' about the database design."

5. **Confirm storage**: After asking Claude to remember something important, ask it to confirm what it stored to verify it was captured correctly.

## How It Works Behind the Scenes

When you use these keywords, Claude identifies them as memory-related commands and calls the appropriate MCP memory service functions:

1. For storage, Claude calls `store_memory` with your content and optional metadata
2. For retrieval, Claude calls functions like `retrieve_memory`, `search_by_tag`, or `recall_memory`
3. For deletion, Claude calls `delete_memory` or `delete_by_tag`

The MCP memory service stores this information in a vector database (ChromaDB) that allows for semantic searching, so you can retrieve information even if you don't use the exact same wording as when you stored it.

## Limitations

- The system works best with explicit commands rather than implicit expectations
- Very long or complex information may be summarized when stored
- Memory retrieval is based on relevance scoring, so results might not always include everything on a topic
- Time-based expressions need to be clear (e.g., "last week" works better than "recently")

## Example Workflow

1. **Store information**: "Please remember that our client meeting is scheduled for Thursday at 2 PM, tag this as 'meetings' and 'client'."

2. **Verify storage**: "What do you know about our upcoming client meeting?"

3. **Later retrieval**: "Search your memory for information tagged 'client' from this week."

4. **Clean up outdated information**: "Delete from memory information about client meetings that happened more than a month ago."