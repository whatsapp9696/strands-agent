#!/usr/bin/env python3
"""
Strands Agent Implementation
Using the strands-agents library for local agent development
"""

from strands import Agent
import time

def create_basic_agent():
    """Create a basic agent using Strands"""
    print("🚀 Creating Strands Agent...")
    
    # Create an agent with default settings
    agent = Agent()
    
    print("✅ Agent created successfully!")
    return agent

def create_custom_agent():
    """Create a customized agent with specific configuration"""
    print("🛠️ Creating Custom Strands Agent...")
    
    # Create an agent with custom settings
    agent = Agent(
        model="gpt-4",  # Specify model if needed
        temperature=0.7,
        max_tokens=1000,
        system_prompt="You are a helpful AI assistant specialized in sentiment analysis and general questions."
    )
    
    print("✅ Custom agent created successfully!")
    return agent

def test_basic_functionality(agent):
    """Test basic agent functionality"""
    print("\n" + "="*50)
    print("🧪 Testing Basic Agent Functionality")
    print("="*50)
    
    test_questions = [
        "Tell me about agentic AI",
        "What is sentiment analysis?",
        "Explain the concept of AI agents in simple terms",
        "How do AI models process natural language?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Test {i}: {question}")
        print("-" * 40)
        
        try:
            # Ask the agent a question
            response = agent(question)
            print(f"🤖 Response: {response}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        # Small delay between requests
        time.sleep(1)

def test_sentiment_analysis(agent):
    """Test sentiment analysis capabilities"""
    print("\n" + "="*50)
    print("🎭 Testing Sentiment Analysis")
    print("="*50)
    
    sentiment_prompt = """
    You are a sentiment analysis expert. For each text I provide, analyze the sentiment and respond with:
    1. Sentiment classification (POSITIVE, NEGATIVE, or NEUTRAL)
    2. Confidence score (0-100%)
    3. Brief reasoning
    
    Format your response as:
    Sentiment: [CLASSIFICATION]
    Confidence: [SCORE]%
    Reasoning: [EXPLANATION]
    """
    
    # Set the system context for sentiment analysis
    agent.system_prompt = sentiment_prompt
    
    test_texts = [
        "I absolutely love this new product! It's amazing!",
        "This movie was really boring and disappointing.",
        "The weather today is okay, nothing special.",
        "I have mixed feelings about this decision - it has pros and cons."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Text {i}: \"{text}\"")
        print("-" * 40)
        
        try:
            response = agent(f"Analyze the sentiment of this text: {text}")
            print(f"📊 Analysis: {response}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        time.sleep(1)

def interactive_mode(agent):
    """Interactive mode for testing the agent"""
    print("\n" + "="*50)
    print("💬 Interactive Agent Mode")
    print("="*50)
    print("Type your questions or 'quit' to exit")
    
    while True:
        user_input = input("\n🧠 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not user_input:
            continue
        
        try:
            print("🤖 Agent: ", end="", flush=True)
            response = agent(user_input)
            print(response)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def create_specialized_agents():
    """Create multiple specialized agents for different tasks"""
    print("\n" + "="*50)
    print("🏭 Creating Specialized Agents")
    print("="*50)
    
    # Sentiment Analysis Agent
    sentiment_agent = Agent(
        system_prompt="""You are a sentiment analysis specialist. Always respond with:
        Sentiment: [POSITIVE/NEGATIVE/NEUTRAL]
        Confidence: [0-100]%
        Reasoning: [brief explanation]"""
    )
    
    # Code Helper Agent
    code_agent = Agent(
        system_prompt="""You are a coding assistant. Provide clear, concise code examples 
        and explanations. Focus on Python programming."""
    )
    
    # General Assistant Agent
    general_agent = Agent(
        system_prompt="""You are a helpful general assistant. Provide clear, 
        informative answers to questions on various topics."""
    )
    
    agents = {
        "sentiment": sentiment_agent,
        "code": code_agent,
        "general": general_agent
    }
    
    print("✅ Created specialized agents:")
    for name in agents.keys():
        print(f"   • {name.title()} Agent")
    
    return agents

def test_specialized_agents(agents):
    """Test the specialized agents"""
    print("\n" + "="*50)
    print("🎯 Testing Specialized Agents")
    print("="*50)
    
    tests = {
        "sentiment": [
            "I'm really excited about this new opportunity!",
            "This service is terrible and unreliable."
        ],
        "code": [
            "Write a Python function to calculate factorial",
            "How do I read a CSV file in Python?"
        ],
        "general": [
            "What is machine learning?",
            "Explain the water cycle"
        ]
    }
    
    for agent_type, test_questions in tests.items():
        print(f"\n🤖 Testing {agent_type.title()} Agent:")
        print("-" * 30)
        
        agent = agents[agent_type]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            try:
                response = agent(question)
                print(f"💡 Response: {response[:200]}..." if len(response) > 200 else f"💡 Response: {response}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
            
            time.sleep(0.5)

def main():
    """Main function to demonstrate Strands agents"""
    print("🧵 Strands Agents Demo")
    print("=" * 30)
    
    try:
        # Create basic agent
        basic_agent = create_basic_agent()
        
        # Test basic functionality
        test_basic_functionality(basic_agent)
        
        # Test sentiment analysis
        test_sentiment_analysis(basic_agent)
        
        # Create specialized agents
        specialized_agents = create_specialized_agents()
        
        # Test specialized agents
        test_specialized_agents(specialized_agents)
        
        # Interactive mode with general agent
        print(f"\n🎮 Starting interactive mode...")
        interactive_mode(specialized_agents["general"])
        
    except ImportError:
        print("❌ Error: strands-agents library not found!")
        print("📦 Please install it with: pip install strands-agents")
        return
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return

if __name__ == "__main__":
    main()