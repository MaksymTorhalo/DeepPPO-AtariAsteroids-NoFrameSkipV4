import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo
import ale_py
import torch
import time
from model import NatureCNN




def evaluate(model_path, record_video=True, num_episodes=3):
    render_mode = "rgb_array" if record_video else "human"
    env = gym.make("ALE/Asteroids-v5",
                   frameskip=1,
                   repeat_action_probability=0.0,
                   obs_type='rgb',
                   render_mode=render_mode)

    if record_video:
        env = RecordVideo(
            env,
            video_folder="./eval_videos",
            name_prefix="ppo_asteroids",
            episode_trigger=lambda x: True  # Record every episode
        )

    env = AtariPreprocessing(env,
                             noop_max=30,
                             frame_skip=4,
                             screen_size=84,
                             terminal_on_life_loss=False,
                             grayscale_obs=True,
                             scale_obs=False)

    env = FrameStackObservation(env, 4)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NatureCNN(env.action_space.n).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        done = False
        total_score = 0
        steps = 0

        print(f"Episode: {episode + 1}")

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.uint8, device=device).unsqueeze(0)
            with torch.no_grad():
                hidden = model(obs_tensor)
                logits = model.policy(hidden)
                action = torch.argmax(logits, dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_score += reward
            steps += 1

            if not record_video:
                time.sleep(0.015)       # for viewable fps

        print(f"Episode {episode + 1} finished | Steps survived: {steps} | Total Score: {total_score}")

    env.close()
    if record_video:
        print("Videos saved to the './eval_videos' directory.")


if __name__ == "__main__":
    weights = "models/final_natureCNN2026_07_10_02_02.pt"
    evaluate(weights, record_video=True, num_episodes=5)