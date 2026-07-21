### 🎾 Tennis Match Analysis with Computer Vision  

## 🏆 Overview  
This repository contains the codebase for a comprehensive **computer vision project** designed for analyzing tennis matches. Leveraging advanced techniques and models, the system provides multi-faceted insights into **gameplay dynamics, player movements, and ball trajectories**.  

## DataSet
Tennis Ball Detection Computer Vision Dataset: https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection


🔗 [Watch the video here](https://github.com/Nader-Mamdouh/FAIR-Talent-Discovery/blob/main/Tennis%20Model.Ai/output_videos/output_video.avi)
---

## 🚀 Key Features  
- **YOLOv8 Player Detection**: Uses **YOLOv8** to detect and track players in tennis match footage.  
- **Fine-Tuning for Ball Detection**: Enhances YOLO to accurately detect the **tennis ball** under different conditions.  
- **Speed Detection**: Computes the speed of both players and the ball for **performance evaluation**.  
- **Mini Court Generation**: Creates a **miniature version of the court**, reflecting real-time player positions and movements.  
- **Key Point Extraction**: Fine-tunes **ResNet50** to extract key court points for **spatial analysis and strategy insights**.  

---

## ⚙️ Usage  
### ▶️ Running the System  
1. Place your **tennis match video** (\`.mp4\` format) inside the \`input_video/\` folder.  
2. Run the main script:  
   \`\`\`sh
   python main.py
   \`\`\`
3. The system will process the video, performing:  
   - Player detection  
   - Ball tracking  
   - Speed calculation  
   - Mini court visualization  
   - Key point extraction  

---

## 🔧 Customization  
- Modify **configuration parameters** in the script to fine-tune the analysis.  
- Train models on **custom datasets** for improved accuracy in different conditions.  

---

## 🤝 Contributions  
Contributions are welcome! Feel free to **open a pull request** for bug fixes, enhancements, or new features.  

📩 For any questions or discussions, reach out via **GitHub Issues**.  

🚀 Happy coding!  " > README.md
