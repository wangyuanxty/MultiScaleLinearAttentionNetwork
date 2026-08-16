# Early prediction of lithium-ion battery degradation with a generative pre-trained transformer 

Received: 17 April 2024
Accepted: 14 November 2025
Published online: 05 December 2025
(I) Check for updates

Jincheng Hu ${ }^{\text {(1) }}{ }^{\text {1,2 }, 5}$, Pengyu Fu ${ }^{\text {(1) }}{ }^{\text {1,2,5 }}$, Zhongbao Wei ${ }^{\text {(1) }}$, Yanjun Huang ${ }^{\text {(1) }}$, Juliana Early ${ }^{\mathbf{4}}$, Ashley Fly ${ }^{\mathbf{2}}$ \& Yuanjian Zhang ${ }^{\boldsymbol{( 1 )}}{ }^{\mathbf{1 , 2}}$

The early detection of degradation in lithium-ion batteries (LIBs) is crucial for effective predictive maintenance and recycling. However, accurately predicting the future degradation of LIBs in early stage is challenging due to the barely noticeable performance changes at initial charging cycles and the long-term nonlinear degradation pattern. In this work, we propose a two-stage earlystage degradation prediction method, BatteryGPT, which employs a Generative Pre-trained Transformer (GPT) to autoregressively predict the charging data of entire lifecycle and a state-of-health (SOH) estimator to correlates the predicted charging data with ageing features in LIBs. The validation demonstrates that BatteryGPT can predict the future LIB degradation with high accuracy using early charging data, before any capacity degradation is evident. Predicting with the first 30\% of the battery lifetime, BatteryGPT significantly outperforms baselines, achieving a root mean square error (RMSE) of 0.213\% for SOH variation prediction, and mean absolute percent errors (MAPE) of 2.30\% and 1.18\% for knee point and EOL predictions. Even predicting with the first 5\% of lifetime charging data, BatteryGPT demonstrates strong early-stage prediction performance.

The extensive application of lithium-ion batteries (LIBs) in storing power generated from sustainable energy sources, such as solar and wind, profoundly impacts the field of energy research and methodologies ${ }^{1}$. This innovation has paved the way for envisioning an energy supply free from fossil fuels, representing a significant stride towards net-zero emissions ${ }^{2}$. However, the performance of LIBs gradually declines over long-term usages due to various factors. For instance, the number of charge-discharge cycles, temperature variations, and efficiency of battery management system (BMS) all impact LIB performance. This negative shift is a long-term feedback delay, primarily manifested in the reduced capacity and increased internal resistance, inducing potential safety risks in emerging sustainable energy systems ${ }^{3}$. If LIB degradation could be detected in the early stages of the lifecycle, sustainable energy systems would enable proactive measures to mitigate these risks and enhance reliability and safety of power usages. Therefore, predicting future LIB degradation based on the initial charging cycles presents opportunities for the safe use and optimization of sustainable energy ${ }^{4}$.

Accurately predicting performance degradation in advance introduces a set of ageing-feature prediction tasks that are crucial to fully unlock the potential of LIBs. These challenges include predicting the state-of-health (SOH) variations ${ }^{5}$, identifying the knee point ${ }^{6}$, and forecasting the end-of-life (EOL) time ${ }^{7}$. SOH describes the current performance of a LIB as a proportion of its initial performance, commonly defined as the ratio of current capacity to initial capacity ${ }^{3}$. Predicting the SOH variations involves analyzing and forecasting

[^0]characteristics changes in battery ageing. This process requires the development of mathematical models and algorithms to track fluctuations in characteristics such as voltage, current, and temperature ${ }^{8}$. These models and algorithms help to identify early ageing patterns of LIB performance degradation, enabling battery management systems (BMSs) to take appropriate actions to optimize the energy usage and prolong the remaining useful life (RUL). The knee point ${ }^{9}$ is a critical moment in LIB performance degradation, typically indicating a time point where the SOH begins to decrease significantly. Monitoring battery capacity and other key indicators can predict when a battery might reach this point, ensuring appropriate action is taken before obvious battery failure ${ }^{10}$ to aid in planning maintenance ${ }^{11}$ and battery replacement ${ }^{12}$. EOL prediction ${ }^{13}$ is a challenging task, which is affected by various factors such as operating environment, charging and discharging patterns, temperature control, and maintenance history ${ }^{14}$. Accurate prediction of EOLs for LIBs significantly enhances their recycling and secondary utilization, especially in critical applications such as medical devices, electric vehicles, and grid energy storage systems ${ }^{15}$. Knee point and EOL predictions are essential in assessing the recycling value of batteries, pinpointing the optimal time for maintenance and reuse to maximize their lifecycle and economic worth ${ }^{16,17}$.

The early prediction of LIBs degradation is significantly challenging due to the long-term nonlinear nature during degradation and the barely noticeable performance changes in the initial charging data, discounting the application of both model-based and data-driven methods. Model-based methods provide an intuitive and scientific description of LIB degradation, stemming from their deep understanding of inherent physical and chemical mechanisms. Electrochemical models ${ }^{18,19}$ are highly significant and accurate due to their precise depiction of battery reaction mechanisms, applied to explain changes in battery capacity and impedance. On the other hand, semiempirical models ${ }^{16,20,21}$, by processing and analyzing large amounts of experimental data, offer a concise and effective method for predicting battery SOH variations, knee points and EOLs. However, these methods may require extensive parameter tuning and experimental validation. The long-term nonlinear changes in performance degradation under different charging and operating conditions pose severe challenges to the application of semi-empirical models. Data-driven methods are much-anticipated solutions, emphasizing the extraction of inherent patterns and hidden features related to performance changes from a wide range of operational features ${ }^{22,23}$. These methods owe their emergence to the rapid advancements in machine learning and deep learning algorithms ${ }^{24}$, which can directly determine performance degradation trends from data without a deep understanding of the internal mechanics.

Existing data-driven solutions typically focus on a limited range of performance prediction, estimating LIB degradation over the next few or dozen charging cycles based on the latest charging data ${ }^{25,26}$, demonstrating commendable accuracy in many applications ${ }^{27}$. However, the effectiveness of data-driven methods largely depends on the quality and relevance of degradation features ${ }^{7}$. Severson et al. ${ }^{4}$ extract features from discharge voltage and capacity curves to predict battery life in early cycles, demonstrating the challenges of nonlinear performance degradation and feature analysis at the EOL for LIBs. Ma et al. ${ }^{28}$ use battery capacity within a specific window as a feature to design a hybrid neural network combining convolutional neural network (CNN) and long short-term memory neural network (LSTM) to estimate capacity in the next window. Cai et al. ${ }^{29}$ propose a CEEMDANTransformer-DNN hybrid model for RUL prediction, suggesting that future research should use various stress features, such as temperature, voltage, and current, as input parameters to achieve more advanced battery performance degradation prediction models due to the significant influence of actual operating conditions and working environments on capacity decay behavior. Xu et al. ${ }^{30}$ utilize Transformer as a global model to predict battery performance degradation, emphasizing the challenges of feature identification in early cycle data and long-term nonlinear prediction, clarifying that without additional corrections, Transformer cannot handle the severe nonlinear degradation at the EOL. Costa et al. ${ }^{31}$ develop ICFormer, a Transformerbased deep learning model that leverages self-attention mechanisms to analyze incremental capacity curves for early knee detection in lithium-ion batteries, highlighting the model's ability to identify degradation patterns and predict battery health indicators more effectively than traditional methods. Zhao et al. ${ }^{32}$ propose a Transformer-embedded model, combining Transformer-based feature extraction with enhanced unscented Kalman filter for SOC estimation, addressing challenges in aging feature identification under varying temperature and operating conditions. Therefore, early charging data often lacks sufficient degradation features, making it difficult for machine learning and deep learning models to recognize performance changes over the entire lifespan of LIBs, limiting their accuracy in predicting long-term degradation in the early stages ${ }^{33}$.

How to capture potential performance changes in future from early charging data is the key to achieve reliable early prediction of LIBs degradation. He et al. ${ }^{34}$ utilize the first 100 cycles to generate complex image features as inputs to a CNN, demonstrating the capability of data-driven methods to uncover early degradation trends. Hsu et al. ${ }^{35}$ directly infer the remaining useful life (RUL) solely from data in the first cycle, further reducing the early data requirements of data-driven approaches. Zhao et al. ${ }^{36}$ utilize voltage, current, and temperature change curves extracted from the first 10 to 110 cycles to construct an input data structure tailored to a multi-branch Vision Transformer model, achieving relatively accurate early lifespan prediction using only 100 cycles of data. However, these methods rely on specific early cycles, leading to reduced prediction accuracy when critical data is missing or the distribution changes ${ }^{37,38}$. Additionally, these methods require relatively complex feature engineering ${ }^{39}$ and often lack intermediate feedback ${ }^{40}$. These limitations pose substantial challenges for large-scale deployment, cross-platform application, and multi-chemistry battery lifetime prediction. To achieve efficient and universal early prediction of battery degradation, more flexible model structures and data-driven strategies must be explored. A feasible solution is to use early charging data to extrapolate future charging behaviours to predict SOH variations of the entire lifecycle. This requires the development of advanced deep learning models capable of handling long-term nonlinear changes and extracting subtle features from early charging data. Indeed, this is a potential application for causal language models. Causal language models ${ }^{41}$ have demonstrated powerful prediction and generation capabilities in Natural Language Processing (NLP). These models learn from vast amounts of text data, enabling to understand and predict the sequential structure of language ${ }^{42}$. Especially with increased model parameters and training data, they can adapt to various task scenarios. In the context of LIB degradation prediction, causal language models can capture and exploit long-term sequential dependencies in charging data, identifying subtle features from early charging cycles to infer and extend potential full lifecycle charging data ${ }^{43}$. This approach may better understand and predict the long-term degradation of LIB performance, providing more accurate information for BMSs, thus enhancing LIB operational efficiency and safety.

In this work, BatteryGPT is developed as an early prediction method for LIB performance degradation, which is capable of accurately predicting lifecycle SOH variations, knee points and EOLs. The method involves a two-stage early prediction strategy. In first stage, a Generative Pre-trained Transformer (GPT) Small Model is employed to autoregressively predict full lifecycle charging data to effectively address the challenge of perceiving long-term nonlinear changes. In the second stage, a SOH estimator in BatteryGPT establishes a direct relationship between the predicted charging data and the future LIB SOH, assisting BatteryGPT in predicting SOH variations, knee points,
and EOLs throughout the full lifecycle of LIB. The 46 battery diagnostic data from the MIT LIB degradation dataset ${ }^{4}$ is employed as training and testing data for BatteryGPT in this work, covering 21,280,768 samples after data processing. In this work, Early Prediction Start Offset (EPSO), defined as the ratio of available battery data to the full lifecycle, establishes a precise starting point for BatteryGPT's predictive analysis within the battery's lifespan. By setting different EPSOs, we quantitatively assess the performance of BatteryGPT. BatteryGPT demonstrates its leading performance in predicting LIB degradation, outperforming baselines. Even with just the first 5\% of charging data (EPSO=5\%), BatteryGPT significantly outperforms baseline models. It improves SOH estimation accuracy by 60.76\%, knee point prediction accuracy by 31.33\%, and EOL prediction accuracy by 4.23\%. This trend of superior performance continues as more data becomes available. When employing the first 30\% of charging cycles (EPSO=30\%), BatteryGPT shows even greater improvements: SOH prediction accuracy improves by 91.80\%, knee point prediction accuracy grows by 92.6\%, and EOL prediction accuracy increases by 54.61\%. This work integrates powerful causal language models into battery ageing modeling, showcasing a valuable research pathway towards predictive maintenance for the sustainable energy storage systems.

## Results

Overview of the methodology
BatteryGPT offers a two-stage pipeline designed for predicting full lifecycle SOH variations to identify the knee point and EOL. The development of BatteryGPT encompasses three main processes: data processing (Fig. 1a,b), battery feature generation (Fig. 1c,d), and degradation prediction (Fig. 1e), with their implementation details available in the Methods section.

During the data processing stage, the EOL is defined (Fig. 1a) when the ratio of the current battery capacity to the initial battery capacity falls below 0.8, which is assumed as the threshold for the unhealthy state. Operational voltage, current, and temperature during battery charging cycles are recorded as the observational charging data for BatteryGPT (Fig. 1a), representing the temporal changes in battery operational status during charging cycles. This charging data may originate from various sensors and sources, potentially with different scales and domains. Through tokenization ${ }^{44}$ (Fig. 1b), the charging data is represented in a consistent numerical space, providing standardized input for the GPT-Small Model ${ }^{45}$ (Fig. 1c). Additionally, converting continuous LIB operational states into token sequences not only effectively compresses the data but also filters potential noise in the observational data, enabling BatteryGPT to identify potential patterns and relationships more efficiently in the battery function sequences.

The GPT-Small Model ${ }^{45}$ (Fig. 1c) in BatteryGPT is a causal language model adept at learning contextual transition patterns within data. In BatteryGPT, the GPT-Small model is trained as an autoregressive deep learning model in a self-supervised learning form (Fig. 1d). Specifically, it predicts the changes in LIB operational states within a certain range based on the context of the tokenized charging data. Through offline training on charging data, the model gains the ability to recognize and infer the evolution of LIB operational states, thereby enabling it to predict future operational characteristics of LIBs. This predictive capability is crucial for understanding and predicting LIB performance degradation, as charging patterns typically exhibit a certain degree of continuity and regularity. Furthermore, as a prototype of an artificial intelligence (AI) foundational model, the GPT-Small model demonstrates generalization capabilities. It can adapt to various battery operational conditions encountered in training data, such as different temperatures and charging scenarios. This adaptability enhances the applicability of the BatteryGPT framework under various use cases and conditions, making it a versatile tool for LIB performance analysis and prediction.

In BatteryGPT, the degradation prediction (Fig. 1e) utilizes a classic data-driven method for SOH estimation, establishing a direct map from LIB state data to SOH variations during the lifecycle ${ }^{3}$. From the predicted SOH variations in the entire lifecycle, we can easily calculate the knee points and EOLs, the details of which can be found in Method section. The SOH estimator within the degradation prediction uses token sequences of LIB operational states within a certain time range as the observational data, developed as a regression predictive deep learning model in a supervised learning format, with current SOH as the label. Specifically, the SOH estimator employs a CNN to learn the latent features of matrix-arranged LIB operational states and is coupled with a LSTM to learn the temporal connections of these latent features with current SOH. The SOH estimator predicts current SOH only based on the current observed LIB operational state, fully leveraging the advantage of data-driven methods in SOH recognition under current battery working conditions. The GPT-Small model and SOH estimator together constitute the two-stage early prediction pipeline (Fig. 1f). First, the GPT-Small model infers full lifecycle battery operational trajectory based on the current operational status. Subsequently, this trajectory is processed through the SOH estimator to calculate the possible full lifecycle SOH variations. This indicates how SOH will change with the charge cycles and clearly identifies the knee point and EOL. This integrated approach in BatteryGPT enables a comprehensive and predictive analysis of LIB degradation over its full lifecycle, providing valuable insights for battery management and maintenance.

## Data overview

Predicting the LIB degradation using early charging data presents challenge in the long-term nonlinear feature detection, primarily because signs of LIB performance decline are subtle in early charging cycles. To capture this behaviour, the MIT LIB performance early degradation dataset ${ }^{4}$, comprising 21,280,768 sample data from 46 commercial phosphate/graphite batteries, is utilized. This dataset, collected under charging currents from 3.7C to 5.9 C, illustrates battery ageing patterns across various charging and discharging scenarios (S.N1).

Figure 2 illustrates the performance of the selected batteries under different charging conditions and degradation trends. In the initial charging cycles, changes in voltage, current, and temperature are consistent under the same conditions. Charging capacities are near its peak, making significant degradation trends elusive. However, after a series of charging cycles, charging capacities begin to decline significantly. In these cycles, parameters such as voltage, current, and temperature start to differ significantly from the initial charging cycles. This indicates an increase in internal impedance of batteries and reveals potential knee points of LIB degradation. The irregular nonlinear performance variations highlight the difficulty in predicting LIB performance degradation from early cycles. Furthermore, despite these LIBs having similar manufacturing conditions, performance variations over the lifecycle suggest that different charging practices can lead to varying degrees of degradation. In addressing concerns about the generalization capability of deep learning models, this work incorporates data from 42 cells, covering a wide array of battery testing conditions such as different temperatures, charge/discharge rates, and cycling protocols. For testing, data from 4 cells under diverse conditions are employed to rigorously evaluate the model's generalization performance, with test conditions intentionally differing from those during training to assess adaptability. To ensure the generalization performance of BatteryGPT and mitigate overfitting, we employed k-fold cross-validation, early stopping, and regularization. K-fold cross-validation partitions the data into multiple folds, allowing training and validation on varied subsets, which enhances model robustness. Early stopping halts the training when the model's performance on a validation set ceases to improve, effectively preventing overfitting to the training data. Moreover, regularization techniques, including dropout and weight decay, are implemented in this work to penalize model complexity for reducing the risk of overfitting.

![](https://cdn.mathpix.com/cropped/979b9c7a-23ac-48e5-a0b7-8711375ffb2d-04.jpg?height=1843&width=1683&top_left_y=193&top_left_x=202)
Fig. 1 | The diagrammatic sketch of BatteryGPT. a is the data collection stage, $L_{n}$ denotes state-of-health (SOH) of the $n$-th charging cycle. $V_{n}, I_{n}, T_{n}$ denotes the voltage, current, temperature during the $n$-th charging cycle, with $C^{m}$ denotes the $m$-th timestep in a charging cycle. b is the tokenization stage. c is the Generative Pre-trained Transformer (GPT) model structure used in this paper. d shows the basic workflow about the autoregressive prediction of the time-series battery features. e shows the estimation of SOH. The SOH estimation is obtained by

minimizing the Mean Squared Error (MSE) between the predicted and true SOH values using several data-driven models, including a Long Short-Term Memory network (LSTM), a Convolutional Neural Network (CNN), and a Multi-Layer Perceptron (MLP). f shows the two-stage pipeline of BatteryGPT, where early-cycle battery data are used to predict the full-lifecycle SOH, End of Life (EOL), and Knee Point.

In addition, the KIT dataset ${ }^{46}$, which exhibits similar degradation characteristics and includes clearly observable knee points and EOL are used to further validate the generalization capability of the proposed method. The experimental results on the KIT dataset are presented in (S.N6), demonstrating consistent predictive performance and supporting the robustness of our approach.

## Early-stage prediction for LIB degradation

In this work, we evaluate the effectiveness of BatteryGPT in predicting LIB performance degradation. specifically using the initial charging cycles. This involves defining the first 30\% of charge cycles as early charging data, with a case study conducted on Battery \#46 to illustrate the practical application. To facilitate this evaluation, we employ

![](https://cdn.mathpix.com/cropped/979b9c7a-23ac-48e5-a0b7-8711375ffb2d-05.jpg?height=1524&width=1685&top_left_y=157&top_left_x=202)
Fig. 2 | Overview about the battery performance degradation in the MIT dataset ${ }^{4}$. Each curve represents a different charging cycle. The blue curves show that the battery charging capacities are close to its initial capacities. In contrast, the red curves indicate that the battery charging capacities are only 80\% of the factory

Figure 3 a illustrates both the actual SOH variations and the corresponding predictions throughout the LIB lifecycles, highlighting the impressive performance from BatteryGPT in early-stage prediction. Despite the challenges posed by imperceptible LIB performance changes in the initial charging data, prediction accuracy improves with the incremental adjustment of EPSOs. Specifically, when EPSO is set to 5\%, the BatteryGPT demonstrates a significant improvement in prediction accuracy after 50\% charging cycles, even after encountering considerable deviations earlier in the lifecycle, and the most accurate prediction is with 30\% EPSO. This trend is further substantiated by Fig. 3e-j, which illustrate the differences between the actual and predicted SOHs at various EPSO levels, thereby confirming the substantial impact of EPSOs on enhancing early prediction capabilities of BatteryGPT.

In predicting the degradation of LIBs, it is crucial to accurately identify the knee point in advance, which delineates two distinct phases of battery ageing. Prior to the knee point, LIB exhibits a linear ageing rate. However, beyond the knee point, LIB degradation accelerates significantly, resulting in a nonlinear and rapid deterioration of LIB capacity. Therefore, in this work, the knee point is defined as the specific charging cycle in the LIB where the SOH variation exhibits the maximum curvature during the lifecycle. We evaluate the effectiveness of BatteryGPT in knee point prediction by calculating the curvature of the predicted SOH variations (Fig. 3b). Initially, EPSO is set to 5\%, the predicted knee point prematurely occurs at 60\% of the lifecycle, deviating by more than 9\% from the actual knee point. A marked decrease in Mean Absolute Error (MAE) is observed, when EPSO is set

![](https://cdn.mathpix.com/cropped/979b9c7a-23ac-48e5-a0b7-8711375ffb2d-06.jpg?height=2013&width=1691&top_left_y=159&top_left_x=197)
Fig. 3 | Performance analysis of BatteryGPT for early-stage prediction of Battery \#46 degradation. a is the prediction results and ground truth of the battery state-of-health (SOH) variations. b is the results of curvature analysis of the predicted SOH trajectories. c shows the cycles and Mean Absolute Error (MAE) for knee point prediction. d shows End of Life (EOL) prediction errors over the charging

Table 1 | Comparison of LIB degradation prediction performance
|  | Method | SOH Prediction RMSE (\%) | SOH Prediction MAPE (\%) | Knee Point Error | Knee Point MAPE (\%) | EOL Error | EOL MAPE (\%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Regressive prediction | CNN | 6.33 | 5.88 | -380 | 67.26 | Out of range | Out of range |
|  | LSTM | 2.61 | 2.04 | -149 | 26.37 | -24 | 2.84 |
|  | CNN-LSTM | 2.63 | 1.90 | -20 | 3.54 | -20 | 2.37 |
| Autoregressive prediction | LSTM | 6.56 | 5.93 | -491 | 86.90 | Out of range | Out of range |
|  | Transformer | 6.49 | 5.73 | -226 | 40.00 | Out of range | Out of range |
|  | BatteryGPT (5) | 2.56 | 1.90 | -177 | 31.33 | -21 | 2.49 |
|  | BatteryGPT (30) | 0.21 | 0.14 | 13 | 2.30 | 10 | 1.18 |


The out of range means that the predicted SOH do not reach 80\%, and EOL can not be obtained.

Moreover, Fig. 3d illustrates the impact of EPSOs on EOL prediction errors. The graph reveals that lower EPSO values lead to substantial deviations from the actual EOL. However, as EPSO increases beyond 15\%, the EOL errors narrow, concentrating within a range of [- 10, 10] cycles. This trend implies that higher EPSO values enhance the accuracy of EOL predictions, with the error gradually diminishing and stabilizing closer to zero.

To demonstrate the superior performance of BatteryGPT, we apply the widely adopted data-driven SOH prediction solutions as baselines. These solutions are categorized into two primary methodologies: regression prediction and autoregressive prediction, with their specific hyperparameters and training configurations outlined in S.N3 and S.N4, respectively. Regression methods (S.N2), including CNNs, LSTM, and ConvLSTM networks, offer cycle-by-cycle predictions of LIB performance degradation. In this work, they are designed to predict LIB performance degradation in the next charge cycle based on the recent three charging cycles only, which contrasts with the early prediction setting of BatteryGPT. In contrast, autoregressive methods (S.N2), including LSTM and Transformer, predict LIB performance degradation of the entire lifecycle based on the charging data from early lifecycle. The SOH estimator for the autoregressive methods are the same as for BatteryGPT. Autoregressive methods establish a high standard for performance with their capability in sequential modeling of time-series data, offering a traditional but advanced benchmark for measuring the prediction efficacy of BatteryGPT.

Table 1 presents evaluation results when EPSO is set to 30\% (BatteryGPT (30)), showcasing impressive early-stage prediction performance of BatteryGPT over the baseline models. In SOH prediction, BatteryGPT (30) achieves an RMSE of just 0.21\% and a MAPE of 0.14\%, significantly leading all baselines. Moreover, BatteryGPT (30) also excels in predicting knee point and EOL, with errors of only 13 cycles and 10 cycles and MAPEs of 2.30\% and 1.18\%, while existing autoregressive baselines such as LSTM and Transformer are unable to provide meaningful predictions for knee points and EOL.

Furthermore, even when compared with regressive baselines that leverage full-cycle data during the entire battery life, BatteryGPT (30) still shows superior performance in all metrics. This comparison highlights the strong generalization and long-term prediction capability of BatteryGPT, even when operating under stricter data constraints. These results clearly demonstrate the practical advantages of the proposed two-stage autoregressive prediction framework.

Compared to baselines, BatteryGPT provides precise degradation predictions by efficiently utilizing early charging data, particularly in predicting the knee point of LIB performance degradation. The performance improvement between BatteryGPT (5) and BatteryGPT (30) further proves that integrating more early data into the prediction activities significantly enhances the predictive capabilities of the model.

## Rationality of the advance performance

The superior performance of BatteryGPT is primarily attributed to its innovative two-stage design for the battery degradation. This pipeline effectively applies the GPT-Small model to autoregressively predict LIB performance variations during the lifecycle, followed by whole degradation process prediction using the SOH estimator. We detail the autoregressive prediction results of the GPT-Small model to demonstrate its effective support for the SOH estimator in the two-stage prediction pipeline.

Figure 4 a demonstrates the absolute errors of the predicted charging cycle features by autoregressive methods. There is a significant deviation in the predictions by LSTM and Transformer, reflecting a failure to capture the underlying ageing patterns of charging data. In contrast, the GPT-Small model exhibits commendable tracking performance for the battery charge cycle data, with its predicted feature changes remaining consistently close to the actual observations. To find the reason, a comparison of the predicted and actual observed trends is conducted with re-sampling from the early, mid, and late phases of a charging cycle (shown in Fig. 4b). LSTM occasionally produces significant prediction distortions, maintaining consistently low or high predicted values at continuous time points, which greatly differ from the actual target values. It has led to a higher concentration of error points around the maximum and minimum errors. On the other hand, the prediction results of Transformer show a trend similar to actual features, but there is a significant prediction bias and potential risk of prediction drift. Notably, during the mid and late phases of this charging cycle, the GPT-Small model accurately predicts the underlying trends of charging cycle features, denoting a robust alignment with the Ground Truth. Despite the observed deviations in the early phase, the GPT-Small model consistently outperforms LSTM and Transformer in terms of overall trend adherence, indicating superior generalization capabilities.

We further investigate the autoregressive prediction results for voltage, current and temperature within the early, mid, and late phases of a charging cycle (shown in Fig. 5). In the early phase of the charging cycle, the baselines, particularly the LSTM, exhibit a broader distribution of error, suggesting a higher variance in prediction accuracy compared to Transformer. The error distribution of the GPT-Small model is notably smaller and more centered around zero, indicating the enhanced prediction precision and consistency. This pattern is consistent across the voltage, current, and temperature predictions, although the improvement in current prediction is less pronounced. Moving to the mid phase of the charging cycle, we observe a similar trend where the GPT-Small model maintains a narrower error distribution, but the distinction between the baselines and the GPT-Small model is less marked. It's noteworthy that the baseline error distribution for temperature prediction is asymmetrical, skewed towards positive errors, suggesting a systematic overestimation bias in the mid phase. In addition, as the charging cycle

![](https://cdn.mathpix.com/cropped/979b9c7a-23ac-48e5-a0b7-8711375ffb2d-08.jpg?height=1088&width=1268&top_left_y=163&top_left_x=410)
Fig. 4 | Analysis of autoregressive prediction results of battery operating features in a charging cycle. a represents the absolute error of battery operating features over 200 timesteps for Long Short-Term Memory network (LSTM), Transformer and Generative Pre-trained Transformer (GPT-Small) model, with the

progresses into its later phases, there is a noticeable increase in both the range and interquartile range (IQR) of errors across all tested methods. This trend is likely due to the inclusion of numerous constant voltage charging cycles in these phases, which introduce complex and challenging battery ageing patterns. Despite these intricacies, the GPT-Small model demonstrates a consistently moderate median error, underscoring its effectiveness even under challenging conditions.

In the charging cycle, the baselines are prone to a wider range of prediction errors, while the GPT-Small model demonstrates a more robust prediction capability across all metrics. The median error lines by the GPT-Small model are consistently closer to zero, and the IQR is smaller, which denotes a higher concentration of predictions around the median with fewer outliers. This suggests that the GPT-Small model in BatteryGPT offers a significant improvement in prediction reliability and accuracy over traditional LSTM and Transformer models for the autoregressive predicting tasks in the two-stage prediction pipeline.

## Discussion

In this work, we introduce BatteryGPT, a promising AI solution designed for the early prediction of LIB performance degradation. It synergies the capabilities of an advanced causal language model with a data-driven SOH estimator, highlighting the potential to revolutionize predictive maintenance and recycling strategies for LIBs. Predicting at the first 5\% cycles (EPSO=5\%), BatteryGPT outshines baselines, showing significant accuracy improvements of 60.76\% in predicting SOH variations. It also achieves a low MAPE of 31.33\% in detecting the knee point and an impressively low MAPE of 2.49\% in identifying the EOL. Furthermore, there is a significant performance enhancement of BatteryGPT for the early prediction when predicting with more charging cycles. By predicting at the first 30\% charging cycles (EPSO=30\%), the accuracy in predicting SOH variations enhances by 91.80\%, knee point prediction improves by 92.02\%, and EOL prediction increases by 52.61\%. BatteryGPT highlights the pivotal role in early prediction of LIB performance degradation and paves an efficient way for capturing the intricate dynamics of LIB ageing through the advanced deep learning models. This presents a riveting prospect for the proactive prediction of LIB performance degradation.

Although previous research indicates that the expansion of datasets and model sizes would lead to significant advancements in GPT-like models for computer vision and natural language processing tasks, this paper faces specific constraints. Due to the limited scale of the dataset available for LIB performance degradation and the restricted computational power at our disposal, we are unable to thoroughly explore the potential enhancements in BatteryGPT performance that could arise from access to more comprehensive battery operating state data and increased model size. However, the current work could be expanded to larger neural network architectures in future research, incorporating extensive data from LIBs in electric vehicles, miniaturized mobile devices, and large-scale energy storage systems. This would evolve BatteryGPT into an AI foundation model with significant application, environmental, and research value.

## Methods

## Tokenization

The charging strategy of the battery management system may lead to variations in the charging voltage $V$ and current $I$, which in turn may lead to fluctuations in the battery temperature $T$. This change in the
operating state of the battery can be described as,

$$
\left\{\begin{array}{c}
S_{i}=\left[v_{i}, c_{i}, t_{i}\right], S=\{V, I, T\}  \tag{1}\\
V=\left[v_{1}, v_{2}, v_{3}, \ldots, v_{i}, \ldots\right] \\
I=\left[c_{1}, c_{2}, c_{3}, \ldots, c_{i}, \ldots\right] \\
T=\left[t_{1}, t_{2}, t_{3}, \ldots, t_{i}, \ldots\right]
\end{array}\right.
$$

where $S$ denotes the set of battery operating features, which includes the time-series charging voltage $V$, the time-series charging current $I$, and the time series $T . v_{i}, c_{i}$ and $t_{i}$ denote the voltage, current and temperature of battery charging at the $i$ moment. In this paper, the battery operating features of the battery charging part are used to characterise the battery operation in the early battery lifecycle, and the autoregressive prediction of entire lifecycle time-series features is achieved by tokenization followed by input to BatteryGPT. The tokenization of the battery voltage, current and temperature data is done to structure the data so that continuous data of different types and ranges can be transformed into a discrete and uniform format, making it easier for BatteryGPT to capture their intrinsic patterns. In addition, introducing a temporal dimension to time-series data can help the model understand the trend of the data over time. In this way, not only can different ranges of values be handled, but data compression can also be achieved to store key information more efficiently. Most importantly, this transformation allows us to directly utilise pre-existing autoregressive model architectures like GPT-2, which can be represented as,

$$
f(V, I, T, \eta)=\left\{\begin{array}{c}
\dot{v}_{l}=\left\lfloor\frac{v_{i}-V_{\text {min }}}{\eta}\right\rfloor, \bar{V}=\left[\dot{v}_{1}, \dot{v}_{2}, \dot{v}_{3}, \ldots, \dot{v}_{l}, \ldots\right],|\bar{V}|=\left\lfloor\frac{V_{\text {max }}-V_{\text {min }}}{\eta}\right\rfloor  \tag{2}\\
\dot{c}_{l}=\left\lfloor\frac{c_{i}-I_{\text {min }}}{\eta}\right\rfloor, \bar{I}=\left[\dot{c}_{1}, \dot{c}_{2}, \dot{c}_{3}, \ldots, \dot{c}_{l}, \ldots\right],|\bar{I}|=\left\lfloor\frac{I_{\text {max }}-I_{\text {min }}}{\eta}\right\rfloor \\
\dot{t}_{l}=\left\lfloor\frac{t_{i}-T_{\text {min }}}{\eta}\right\rfloor, \bar{T}=\left[\dot{t}_{1}, \dot{t}_{2}, \dot{t}_{3}, \ldots, \dot{t}_{l}, \ldots\right],|\bar{T}|=\left\lfloor\frac{T_{\text {max }}-T_{\text {min }}}{\eta}\right\rfloor
\end{array}\right.
$$

where $f$ is the discretiser constructed in this paper, which generates the discretised representation of the battery operating trajectory. Vmin, Imin and Tmin are the minimum bounds of charging voltage, current and temperature, and Vmax, Imax and Tmax are the theoretical maximum bounds of battery charging voltage, current and temperature. $\eta$ is the discrete step size of the battery operating state, which is set to 0.025 in this paper, and the value is used to characterise the discrete features of the battery operating features while still retaining the change trend. $\bar{V}, \bar{I}$ and $\bar{T}$ represent the discretised charging voltage, current and temperature features respectively. $|\bar{V}|,|\bar{I}|$ and $|\bar{T}|$ denote the state discrete space of charging voltage, current and temperature. After state discretisation, the battery operating features can be expressed as,

$$
\begin{equation*}
\dot{s}_{l}=\left[\dot{v}_{l}, \dot{c}_{l}, \dot{t}_{l}\right], \bar{S}=\left[\dot{s}_{1}, \dot{s}_{2}, \dot{s}_{3}, \ldots, \dot{s}_{l}, \ldots\right],|\bar{S}|=|\bar{V}|+|\bar{I}|+|\bar{T}| \tag{3}
\end{equation*}
$$

Where $\bar{S}$ denotes the battery operating features after tokenization and $s_{i}$ denotes the battery feature token at moment $i$, which contains the current operating voltage, current and temperature tokens.

The temporal dimension is implemented through the natural ordering of the continuous input sequence, working together with positional embedding to help the model decode the temporal and sequential structure of the input data without any feature engineering, as all parameters are learned directly from the data, as shown in Supplementary Fig. S1.

## GPT-small model in batteryGPT

In this paper, a refined GPT model (GPT-Small) is used in BatteryGPT to perform autoregressive prediction of time-series battery operating features, i.e., an attempt to predict the next value of a sequence based

![](https://cdn.mathpix.com/cropped/979b9c7a-23ac-48e5-a0b7-8711375ffb2d-09.jpg?height=959&width=837&top_left_y=165&top_left_x=1090)
Fig. 5 | Error distributions of autoregressive prediction results of battery operating features across different phases (Early, Mid, Late) in a charging cycle. T, I, and V represent temperature, current, and voltage, respectively. The error distributions of Long Short-Term Memory network (LSTM) and Transformer are depicted as violin plots in the pink area, while the error distributions of Generative Pre-trained Transformer (GPT-Small) model are shown in the deep blue area, with the median indicated by the black line and the interquartile range (IQR) indicated by the red line. Source data are provided as a Source Data file.

on its previous value, which can be represented as,

$$
\begin{equation*}
\left[\dot{s}_{l+1}, s_{l+2}, s_{l+3}, \ldots, s_{l+n}\right]=G_{\text {battery }}\left(\dot{s_{1}}, \dot{s_{2}}, \dot{s_{3}}, \ldots, \dot{s_{l}}\right) \tag{4}
\end{equation*}
$$

where $n$ is the self-referential prediction time domain and $G_{\text {battery }}$ is the GPT neural network used by BatteryGPT. Compared with the opensource GPT-2 model, $G_{\text {battery }}$ 's size of 42M is 33.87\% of the smallest size GPT-2. It supports a context size of up to 256, can input 384 feature channels, and employs 24 layers of Transformer Decoder Blocks, as shown in Supplementary Fig. S3. The embedding module of $G_{\text {battery }}$ first converts the battery work tokens into vectors, the process can be represented as follows,

$$
\begin{equation*}
E\left(\dot{s}_{l}\right)=E_{\text {feature }}\left(\dot{s}_{l}\right)+E_{\text {position }}(i) \tag{5}
\end{equation*}
$$

where $E\left(s_{i}\right)$ is the embedding representation of the $i$ th token, which is derived from the summation of two components, i.e., the battery feature embedding $E_{\text {feature }}\left(s_{i}\right)$ and the positional embedding $E_{\text {position }}(i) . E_{\text {feature }}\left(s_{i}\right)$ is Each possible token is assigned a 384-dimensional vector and $E_{\text {position }}(i)$ denotes the position of the token in the sequence. In the $G_{\text {battery }}$ setting, the position embedding is added to the embedding representation to ensure that the model can take into account the order of the tokens in the sequence. This embedding sequence representing the timing features of the battery operating features is fed into the 24-layer Transformer Decoder Blocks, which can be represented as,

$$
\left\{\begin{array}{c}
h_{0}=E(X)  \tag{6}\\
h_{l}=\operatorname{Block}_{l}\left(h_{l-1}\right)
\end{array}\right.
$$

For each position $i$ in the output of the last layer, we compute a probability distribution:

$$
\begin{equation*}
P_{i}\left(\dot{s}_{l}\right)=\operatorname{softmax}\left(W_{\text {output }} \cdot h_{24, i}+b_{\text {output }}\right) \tag{7}
\end{equation*}
$$

where $P_{i}$ is a vector of vocab size, representing the probability distribution of the next token after position $i$. For the data studied in this paper, the calculation result for the vocab size is 743. Based on the above probability distribution, the next token is selected or sampled. In this way, the output of the embedding layer directly becomes the input of the Transformer module, and then the output of the Transformer module is transformed into the probability distribution of the token so that the next token can be selected.This is the relationship between the embedding layer and the Transformer module, and is reflected in the above equation.

The self-attention mechanisms and the autoregressive prediction process, which forms the foundation of our method to obtain the complete battery trajectory over all cycles, are shown in Supplementary Fig. S4.

## Training Setting of BatteryGPT

Hyperparameter Optimization of GPT-Small model is detailed in S.N5. In the BatteryGPT, the goal of the GPT-Small model is to predict the next token based on the current and previous tokens, enabling our method to start predictions from any cycle in the battery aging process. Therefore, a commonly used loss function is the cross-entropy loss. Specifically, we will use the model's output probability distribution to calculate the cross-entropy loss at each position and average over the entire sequence. Let the probability distribution predicted by the model for position $i$ be, and the true token be $\left(s_{i}\right)$, then the crossentropy loss for that position is:

$$
\begin{equation*}
\mathrm{CE}(t)=-\log P_{t}\left(x_{t}\right) \tag{8}
\end{equation*}
$$

where is the probability that the model is assigned for the real token at position $t$. The total cross-entropy loss is the average of the losses at all positions:

$$
\begin{equation*}
\operatorname{Loss}=\frac{1}{L} \sum_{t=1}^{L} \mathrm{CE}(t) \tag{9}
\end{equation*}
$$

where $L$ is the length of the sequence and the value is set to 256 in training. In training, the samples from the MIT Early Degradation dataset are divided into a training set and a validation set. The first 42 battery samples are used as the training set and the rest as the validation set. The training is terminated when the autoregressive error, root mean square error, of the validation and training sets is less than 5\% or the epoch number reaches 500000. The minimum epoch number is set to 1000 and the initialised learning rate is 0.001, rising with epoch to 0.0001 . All input samples are divided into windows of length 256 with a batch size of 64 . In this work, BatteryGPT are trained based on a single NVIDIA RTX4090.

## Identify the knee point and EOL from SOH variations

Knee point is commonly used to identify a significant turning point in a curve. In this paper, the point of maximum curvature is used as the knee point. The formula for calculating curvature is as follows:

$$
\begin{equation*}
\kappa=\frac{\left|y^{\prime \prime}\right|}{\left(1+y^{\prime 2}\right)^{\frac{3}{2}}} \tag{10}
\end{equation*}
$$

where $y=f(x)$ is the equation of SOH curve, and $y^{\prime}$ and $y^{\prime \prime}$ are the first and second derivatives of the equation, respectively.

EOL typically refers to the point at which the battery reaches the end of its usable life, at which time it can no longer effectively store and release energy. In this paper, EOL is defined as the number of cycles after which the capacity falls below 80\% of its initial capacity.

## Data-driven method for SOH estimator

BatteryGPT uses advanced and widely used data-driven methods to generate reliable SOH estimates and test baselines. In this study, we use the RMSE, Absolute Error (AE), and MAPE to evaluate and give a baseline for SOH estimation, knee point, and remaining lifetime prediction:

$$
\left\{\begin{array}{c}
\mathrm{RMSE}=\sqrt{\frac{\sum_{i=1}^{M}\left(y_{i}-\hat{y}_{i}\right)^{2}}{M}}  \tag{11}\\
\mathrm{AE}_{i}=y_{i}-\hat{y}_{i} \mid \\
\mathrm{MAPE}=\frac{100 \%}{n} \sum_{i=1}^{n}\left|\frac{y_{i}-\hat{y}_{i}}{y_{i}}\right|
\end{array}\right.
$$

where $y_{i}$ and $\hat{y}_{i}$ are the measured and final estimates of SOH, knee point or remaining lifetime. $M$ represents the total number of samples of interest.

## Data availability

The datasets used in this study are available at https://data.matr.io/1. All data used in this study can be accessed freely without restrictions. Source data are provided with this paper.

## Code availability

The code is available from the GitHub link at (https://github.com/ ReparkHjc/BatteryGPT,) and has been archived on Zenodo with the https://doi.org/10.5281/zenodo. $17420674{ }^{47}$ under the MIT License.

## References

1. Sun, X., Ouyang, M. \& Hao, H. Surging lithium price will not impede the electric vehicle boom. Joule 6, 1738-1742 (2022).
2. Moving forward with batteries. Nat. Sustain. 6, 721-722 (2023).
3. Roman, D., Saxena, S., Robu, V., Pecht, M. \& Flynn, D. Machine learning pipeline for battery state-of-health estimation. Nat. Mach. Intell. 3, 447-456 (2021).
4. Severson, K. A. et al. Data-driven prediction of battery cycle life before capacity degradation. Nat. Energy 4, 383-391 (2019).
5. Ng, M.-F., Zhao, J., Yan, Q., Conduit, G. J. \& Seh, Z. W. Predicting the state of charge and health of batteries using data-driven machine learning. Nat. Mach. Intell. 2, 161-170 (2020).
6. Sohn, S., Byun, H.-E. \& Lee, J. H. Two-stage deep learning for online prediction of knee-point in Li-ion battery capacity degradation. Appl. Energy 328, 120204 (2022).
7. Fei, Z., Yang, F., Tsui, K.-L., Li, L. \& Zhang, Z. Early prediction of battery lifetime via a machine learning based framework. Energy 225, 120205 (2021).
8. Tian, H., Qin, P., Li, K. \& Zhao, Z. A review of the state of health for lithium-ion batteries: research status and suggestions. J. Clean. Prod. 261, 120813 (2020).
9. Liu, K., Tang, X., Teodorescu, R., Gao, F. \& Meng, J. Future ageing trajectory prediction for lithium-ion battery considering the knee point effect. IEEE Trans. Energy Convers. 37, 1282-1291 (2021).
10. Chen, J. C., Chen, T.-L., Liu, W.-J., Cheng, C. \& Li, M.-G. Combining empirical mode decomposition and deep recurrent neural networks for predictive maintenance of lithium-ion battery. Adv. Eng. Inform. 50, 101405 (2021).
11. von Bülow, F. \& Meisen, T. A review on methods for state of health forecasting of lithium-ion batteries applicable in real-world operational conditions. J. Energy Storage 57, 105978 (2023).
12. Ren, H., Zhao, Y., Chen, S. \& Wang, T. Design and implementation of a battery management system with active charge balance based on the SOC and SOH online estimation. Energy 166, 908-917 (2019).

13. Meng, J., Yue, M., \& Diallo, D. A degradation empirical-model-free battery end-of-life prediction framework based on Gaussian process regression and Kalman filter. IEEE Transactions on Transportation Electrification (2022).
14. Sulzer, V. et al. The challenge and opportunity of battery lifetime prediction from field data. Joule 5, 1934-1955 (2021).
15. Chen, M. et al. Recycling end-of-life electric vehicle lithium-ion batteries. Joule 3, 2622-2646 (2019).
16. Zhu, J. et al. End-of-life or second-life options for retired electric vehicle batteries. Cell Rep. Phys. Sci. 2, 100537 (2021).
17. Fallah, N. \& Fitzpatrick, C. Exploring the state of health of electric vehicle batteries at end of use; hierarchical waste flow analysis to determine the recycling and reuse potential. J. Remanufacturing 14, 155-168 (2024).
18. Xiong, R., Li, L., Li, Z., Yu, Q. \& Mu, H. An electrochemical model based degradation state identification method of lithium-ion battery for all-climate electric vehicles application. Appl. Energy 219, 264-275 (2018).
19. Barzacchi, L., Lagnoni, M., Di Rienzo, R., Bertei, A. \& Baronti, F. Enabling early detection of lithium-ion battery degradation by linking electrochemical properties to equivalent circuit model parameters. J. Energy Storage 50, 104213 (2022).
20. Pelletier, S., Jabali, O., Laporte, G. \& Veneroni, M. Battery degradation and behaviour for electric vehicles: review and numerical analyses of several models. Transportation Res. Part B: Methodol. 103, 158-187 (2017).
21. Thingvad, A., Calearo, L., Andersen, P. B. \& Marinelli, M. Empirical capacity measurements of electric vehicles subject to battery degradation from V2G services. IEEE Trans. Vehicular Technol. 70, 7547-7557 (2021).
22. Liu, X. et al. A generalizable, data-driven online approach to forecast capacity degradation trajectory of lithium batteries. J. Energy Chem. 68, 548-555 (2022).
23. Xiong, R. et al. A data-driven method for extracting aging features to accurately predict the battery health. Energy Storage Mater. 57, 460-470 (2023).
24. Berecibar, M. Machine-learning techniques used to accurately predict battery life. Nature 568, 325-326 (2019).
25. Zhang, Y. et al. Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning. Nat. Commun. 11, 1706 (2020).
26. Strange, C. \& Dos Reis, G. Prediction of future capacity and internal resistance of Li-ion cells from one cycle of input data. Energy AI 5, 100097 (2021).
27. Sulzer, V., Mohtat, P., Lee, S., Siegel, J. B. \& Stefanopoulou, A. G. Promise and challenges of a data-driven approach for battery lifetime prognostics. In 2021 American Control Conference (ACC), 4427-4433 (IEEE, 2021).
28. Ma, G. et al. Remaining useful life prediction of lithium-ion batteries based on false nearest neighbors and a hybrid neural network. Appl. Energy 253, 113626 (2019).
29. Cai, Y. et al. Early prediction of remaining useful life for lithium-ion batteries based on ceemdan-transformer-dnn hybrid model. Heliyon 9, e17754 (2023).
30. Xu, R., Wang, Y. \& Chen, Z. A hybrid approach to predict battery health combined with attention-based transformer and online correction. J. Energy Storage 65, 107365 (2023).
31. Costa, N., Anseán, D., Dubarry, M. \& Sánchez, L. Icformer: A deep learning model for informed lithium-ion battery diagnosis and early knee detection. J. Power Sources 592, 233910 (2024).
32. Zhao, S.-Y. et al. A novel transformer-embedded lithium-ion battery model for joint estimation of state-of-charge and state-of-health. Rare Met. 43, 5637-5651 (2024).
33. Shen, S., Liu, B., Zhang, K. \& Ci, S. Toward fast and accurate soh prediction for lithium-ion batteries. IEEE Trans. Energy Convers. 36, 2036-2046 (2021).
34. He, N., Wang, Q., Lu, Z., Chai, Y. \& Yang, F. Early prediction of battery lifetime based on graphical features and convolutional neural networks. Appl. Energy 353, 122048 (2024).
35. Hsu, C.-W., Xiong, R., Chen, N.-Y., Li, J. \& Tsou, N.-T. Deep neural network battery life and voltage prediction by using data of one cycle only. Appl. Energy 306, 118134 (2022).
36. Zhao, W., Ding, W., Zhang, S. \& Zhang, Z. Enhancing lithium-ion battery lifespan early prediction using a multi-branch vision transformer model. Energy 131816 (2024).
37. Yang, A., Wang, Y., Tsui, K. L. \& Zi, Y. Lithium-ion battery soh estimation and fault diagnosis with missing data. In 2019 IEEE International Instrumentation and Measurement Technology Conference (I2MTC), 1-6 (IEEE, 2019).
38. Demirci, O., Taskin, S., Schaltz, E. \& Demirci, B. A. Review of battery state estimation methods for electric vehicles-part ii: Soh estimation. J. Energy Storage 96, 112703 (2024).
39. Yang, M. et al. Predict the lifetime of lithium-ion batteries using early cycles: a review. Appl. Energy 376, 124171 (2024).
40. Yao, Z. et al. Machine learning for a sustainable energy future. Nat. Rev. Mater. 8, 202-215 (2023).
41. Radford, A. et al. Improving language understanding by generative pre-training. OpenAI (2018).
42. Yang, Y., Li, Y. \& Quan, X. Ubar: Towards fully end-to-end taskoriented dialog system with gpt-2. Proc. AAAI Conf. Artif. Intell. 35, 14230-14238 (2021).
43. Yu, B., Li, Y. \& Wang, J. Detecting causal language use in science findings. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 4664-4674 (2019).
44. Hofmann, V., Schütze, H. \& Pierrehumbert, J. B. An embarrassingly simple method to mitigate undesirable properties of pretrained language model tokenizers (Association for Computational Linguistics, 2022).
45. Paaß, G. \& Giesselbach, S. Foundation Models for Natural Language Processing: Pre-trained Language Models Integrating Media (Springer Nature, 2023).
46. Luh, M. \& Blank, T. Comprehensive battery aging dataset: capacity and impedance fade measurements of a lithium-ion nmc/c-sio cell. Sci. Data 11, 1004 (2024).
47. ReparkHjc \& ilm75862. Reparkhjc/batterygpt: test-1 https://doi.org/ 10.5281/zenodo. 17420675 (2025).

## Acknowledgements

This work was supported by the National Key R\&D Program of China (Grant No. 2024YFB2505601, Y. Z.), Joint Fund for Innovative Enterprise Development (U23B2O61, Y. H.) and the Xiaomi Foundation. The authors also thank Prof. Harry Hoster and Dr. Yi Li for their valuable support and helpful discussions that contributed to improving the quality of this paper.

## Author contributions

J.H. designed the study concept and methodology, and manuscript drafting. P.F. developed the software, and contributed to data validation and visualization. Z.W. performed the formal analysis and contributed to data interpretation. Y.H. provided resources and supervised the project. J.E. managed project administration. A.F. reviewed and edited the manuscript. Y.Z. acquired funding, supervised the research, and handled correspondence.

## Competing interests

The authors declare no competing interests.

## Additional information

Supplementary information The online version contains supplementary material available at https://doi.org/10.1038/s41467-025-66819-0.

Correspondence and requests for materials should be addressed to Yanjun Huang or Yuanjian Zhang.

Peer review information Nature Communications thanks Nahuel Costa, and the other, anonymous, reviewer(s) for their contribution to the peer review of this work. A peer review file is available.

Reprints and permissions information is available at http://www.nature.com/reprints

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Open Access This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You do not have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party material in this article are included in the article's Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by-nc-nd/4.0/.
© The Author(s) 2025


[^0]:    ${ }^{1}$ School of Automotive Studies, Tongji University, Shanghai, China. ${ }^{2}$ Department of Aeronautical and Automotive Engineering, Loughborough University, Loughborough, United Kingdom. ${ }^{3}$ National Engineering Research Centre for Electric Vehicles, Beijing Institute of Technology, Beijing, China. ${ }^{4}$ School of Mechanical and Aerospace Engineering, Queen's University Belfast, Northern Ireland, UK. ${ }^{5}$ These authors contributed equally: Jincheng Hu, Pengyu Fu. e-mail: yanjun_huang@tongji.edu.cn; Y.Y.Zhang@lboro.ac.uk

