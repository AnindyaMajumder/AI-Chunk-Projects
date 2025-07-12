# Benji AI Assistant - Django Integration Guide

## Overview

This project provides **Benji**, an AI-powered insurance claims assistant built with LangChain, designed for seamless integration with Django backend applications. Benji processes insurance documents and provides intelligent, context-aware responses to user queries.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Django Web    │────│   app.py        │────│   LangChain     │
│   Application   │    │   (This Module) │    │   + OpenAI      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  User Sessions  │    │  FAISS Vector   │    │   PDF Documents │
│  & Memory       │    │  Store          │    │   + Training    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Document Setup

Place your documents in the `data/` directory:
- PDF files (insurance documents, policies, etc.)
- `Training Phrases.csv` (FAQ and training data)

### 4. Django Integration

#### Basic Integration

```python
# In your Django views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import sys
import os

# Add the AI project to Python path
sys.path.append('/path/to/your/AI-Learning')
from app import get_benji_response, create_session_memory

@csrf_exempt
def chat_api(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        question = data.get('question', '')
        
        # Get or create session memory
        if 'benji_memory' not in request.session:
            request.session['benji_memory'] = create_session_memory()
        
        memory = request.session['benji_memory']
        response = get_benji_response(question, memory)
        
        return JsonResponse({
            'response': response,
            'status': 'success'
        })
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)
```

#### Advanced Integration with User Management

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
import pickle
import base64

class BenjiSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    session_key = models.CharField(max_length=40)
    memory_data = models.TextField()  # Serialized memory
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

# views.py
from django.contrib.auth.decorators import login_required
from langchain.memory import ConversationBufferMemory
import pickle
import base64

@login_required
@csrf_exempt
def chat_with_benji(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        question = data.get('question', '')
        
        # Get or create user-specific session
        benji_session, created = BenjiSession.objects.get_or_create(
            user=request.user,
            session_key=request.session.session_key,
            defaults={'memory_data': ''}
        )
        
        # Load memory
        if benji_session.memory_data:
            memory_bytes = base64.b64decode(benji_session.memory_data)
            memory = pickle.loads(memory_bytes)
        else:
            memory = create_session_memory()
        
        # Get response
        response = get_benji_response(question, memory)
        
        # Save memory
        memory_bytes = pickle.dumps(memory)
        benji_session.memory_data = base64.b64encode(memory_bytes).decode()
        benji_session.save()
        
        return JsonResponse({
            'response': response,
            'status': 'success',
            'user': request.user.username
        })
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)
```

### 5. URL Configuration

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/benji/', views.chat_with_benji, name='chat_with_benji'),
]
```

## API Reference

### Functions Available for Django

#### `get_benji_response(question, session_memory=None)`

Get a response from Benji AI.

**Parameters:**
- `question` (str): User's question or query
- `session_memory` (ConversationBufferMemory, optional): Session-specific conversation memory

**Returns:**
- `str`: Benji's response

**Example:**
```python
response = get_benji_response("What is covered under fire insurance?")
```

#### `create_session_memory()`

Create a new conversation memory for a user session.

**Returns:**
- `ConversationBufferMemory`: New memory instance

**Example:**
```python
memory = create_session_memory()
response = get_benji_response("Hello", memory)
```

#### `initialize_benji_chain()`

Initialize the complete Benji AI chain (usually called automatically).

**Returns:**
- `ConversationalRetrievalChain`: Configured chain ready for use
