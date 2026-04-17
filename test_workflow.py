#!/usr/bin/env python
"""Test workflow execution end-to-end."""
import asyncio
import json
import httpx
import time

async def main():
    base_url = "http://localhost:8000/api/v1"
    
    async with httpx.AsyncClient() as client:
        # 1. Create workflow
        print("=" * 60)
        print("1️⃣  CREATING WORKFLOW...")
        print("=" * 60)
        response = await client.post(
            f"{base_url}/workflows",
            json={"name": "E2E Test", "goal": "Explain the future of AI"},
        )
        wf_data = response.json()
        wf_id = wf_data["workflow_id"]
        print(f"✅ Workflow Created: {wf_id}")
        print(f"   Initial Status: {wf_data['status']}")
        
        # 2. Wait for execution
        print(f"\n⏳ Waiting 35 seconds for agents to execute...")
        await asyncio.sleep(35)
        
        # 3. Fetch final status
        print(f"\n2️⃣  CHECKING FINAL STATUS...")
        print("=" * 60)
        response = await client.get(f"{base_url}/workflows/{wf_id}")
        final = response.json()
        
        print(f"✅ FINAL STATUS: {final['status'].upper()}")
        print(f"   Output Length: {len(final.get('final_output', ''))} characters")
        print(f"   Created: {final['created_at']}")
        print(f"   Updated: {final['updated_at']}")
        
        if final.get('final_output'):
            print(f"\n3️⃣  OUTPUT PREVIEW (first 800 chars):")
            print("=" * 60)
            print(final['final_output'][:800])
            print("...\n")
        
        # 4. Get agent details
        print(f"4️⃣  AGENT EXECUTION DETAILS:")
        print("=" * 60)
        response = await client.get(f"{base_url}/workflows/{wf_id}/agents")
        agents = response.json()
        for agent in agents:
            print(f"   {agent['emoji']} {agent['role']}: {agent['status']}")
        
        print(f"\n✅ WORKFLOW EXECUTION COMPLETE!")

if __name__ == "__main__":
    asyncio.run(main())
