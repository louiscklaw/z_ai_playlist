from openai import OpenAI

client = OpenAI(
    api_key="00c4671f20784e868004b670aa3e8a13.fm69RbbVi1yFllIN",
    base_url="https://api.z.ai/api/paas/v4/",
)

completion = client.chat.completions.create(
    model="GLM-4.5-Flash",
    messages=[
        {"role": "system", "content": "You are a smart and creative novelist"},
        {
            "role": "user",
            "content": "Please write a short fairy tale story as a fairy tale master",
        },
    ],
)

print(completion.choices[0].message.content)
