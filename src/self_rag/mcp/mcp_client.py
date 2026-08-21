import asyncio
import json

from mcp import ClientSession
from mcp.client.sse import sse_client
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

SERVER_SSE_URL = "http://127.0.0.1:8000/sse"
console = Console()


def display_banner():
    console.print(
        Panel.fit(
            "[bold cyan]Self-RAG MCP Interactive Client[/bold cyan]\n"
            "[dim]Connected via SSE Transport[/dim]",
            border_style="cyan",
        )
    )


def print_tool_response(response, title: str):
    """Parses and renders tool content output prettily based on data type."""
    for content in response.content:
        text_data = content.text
        try:
            # Try parsing structured JSON payload
            parsed = json.loads(text_data)
            
            # If the output contains a markdown answer field, render as Rich Markdown
            if isinstance(parsed, dict) and "result" in parsed and isinstance(parsed["result"], dict):
                answer_text = parsed["result"].get("answer", str(parsed["result"]))
                console.print(Panel(Markdown(answer_text), title=f"[bold green]{title}[/bold green]", border_style="green"))
            else:
                # Fallback to pretty formatted JSON
                formatted_json = json.dumps(parsed, indent=2)
                console.print(Panel(formatted_json, title=f"[bold blue]{title}[/bold blue]", border_style="blue"))
        except (json.JSONDecodeError, TypeError):
            # Fallback to standard Markdown/Text rendering
            console.print(Panel(Markdown(text_data), title=f"[bold green]{title}[/bold green]", border_style="green"))


async def show_tools(session: ClientSession):
    """Fetches and displays available tools in a formatted table."""
    with console.status("[bold yellow]Discovering server tools...[/bold yellow]"):
        tools_response = await session.list_tools()

    table = Table(title="Available Server Tools", show_header=True, header_style="bold magenta")
    table.add_column("Tool Name", style="cyan", width=18)
    table.add_column("Description", style="white")

    for tool in tools_response.tools:
        table.add_row(tool.name, tool.description.strip())

    console.print(table)
    console.print()


async def handle_rag_answer(session: ClientSession):
    question = Prompt.ask("\n[bold yellow]Enter your question[/bold yellow]")
    if not question.strip():
        console.print("[red]Question cannot be empty.[/red]")
        return

    retries = IntPrompt.ask("[yellow]Max retries[/yellow]", default=2)

    with console.status("[bold green]Running Self-RAG state graph...[/bold green]", spinner="dots"):
        response = await session.call_tool(
            "rag_answer",
            arguments={"question": question, "max_retries": retries},
        )
    
    print_tool_response(response, "Self-RAG Response")


async def handle_retrieve(session: ClientSession):
    query = Prompt.ask("\n[bold yellow]Enter retrieval search query[/bold yellow]")
    if not query.strip():
        console.print("[red]Query cannot be empty.[/red]")
        return

    top_k = IntPrompt.ask("[yellow]Number of document chunks (top_k)[/yellow]", default=3)

    with console.status("[bold green]Retrieving chunks from vector store...[/bold green]", spinner="dots"):
        response = await session.call_tool(
            "retrieve",
            arguments={"query": query, "top_k": top_k},
        )

    print_tool_response(response, "Retrieved Documents")


async def handle_health(session: ClientSession):
    with console.status("[bold green]Checking server health...[/bold green]", spinner="dots"):
        response = await session.call_tool("server_health", {})

    print_tool_response(response, "Server Diagnostics")


async def main():
    console.clear()
    display_banner()

    try:
        async with sse_client(SERVER_SSE_URL) as (read, write):
            async with ClientSession(read, write) as session:
                with console.status(f"[bold cyan]Connecting to {SERVER_SSE_URL}...[/bold cyan]", spinner="dots"):
                    await session.initialize()
                console.print("[bold green]✓ Successfully connected to Self-RAG MCP Server![/bold green]\n")

                await show_tools(session)

                while True:
                    console.print("\n[bold underline]Select an action:[/bold underline]")
                    console.print("[1] 💬 Ask Question ([cyan]rag_answer[/cyan])")
                    console.print("[2] 🔍 Raw Search ([cyan]retrieve[/cyan])")
                    console.print("[3] 🏥 System Health ([cyan]server_health[/cyan])")
                    console.print("[4] 📋 List Tools")
                    console.print("[0] 🚪 Exit")

                    choice = Prompt.ask("[bold yellow]Choice[/bold yellow]", choices=["0", "1", "2", "3", "4"], default="1")

                    if choice == "1":
                        await handle_rag_answer(session)
                    elif choice == "2":
                        await handle_retrieve(session)
                    elif choice == "3":
                        await handle_health(session)
                    elif choice == "4":
                        await show_tools(session)
                    elif choice == "0":
                        console.print("\n[bold cyan]Goodbye![/bold cyan]")
                        break

    except ConnectionRefusedError:
        console.print(f"[bold red]Error:[/bold red] Could not connect to [yellow]{SERVER_SSE_URL}[/yellow].")
        console.print("[dim]Ensure your server is running with `transport='sse'`.[/dim]")
    except Exception as e:
        console.print(f"[bold red]Unexpected Error:[/bold red] {e}")


if __name__ == "__main__":
    asyncio.run(main())