# In[ ]:
# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# In[ ]:
# files
file_array = []
dir_path = ""
a3_feat_300 = pd.read_csv(f"{dir_path}3Feature_100200300_Output.csv")
a3_feat_75 = pd.read_csv(f"{dir_path}3Feature_255075_Output.csv")
a3_feat_120 = pd.read_csv(f"{dir_path}3Feature_50100150_Output.csv")
a767_band_300 = pd.read_csv(f"{dir_path}767Bands_100200300_Output.csv")
a767_band_75 = pd.read_csv(f"{dir_path}767Bands_255075_Output.csv")
a767_band_120 = pd.read_csv(f"{dir_path}767Bands_50100150_Output.csv")
alit_band_300 = pd.read_csv(f"{dir_path}LitBands61_100200300_Output.csv")
alit_band_75 = pd.read_csv(f"{dir_path}LitBands61_255075_Output.csv")
alit_band_120 = pd.read_csv(f"{dir_path}LitBands61_50100150_Output.csv")
alit_inter_300 = pd.read_csv(f"{dir_path}LitIntersection_100200300_Output.csv")
alit_inter_75 = pd.read_csv(f"{dir_path}LitIntersection_255075_Output.csv")
alit_inter_120 = pd.read_csv(f"{dir_path}LitIntersection_50100150_Output.csv")

file_array.append(a767_band_300)
file_array.append(a767_band_120)
file_array.append(a767_band_75)
file_array.append(a3_feat_300)
file_array.append(a3_feat_120)
file_array.append(a3_feat_75)
file_array.append(alit_inter_300)
file_array.append(alit_inter_120)
file_array.append(alit_inter_75)
file_array.append(alit_band_300)
file_array.append(alit_band_120)
file_array.append(alit_band_75)

# In[ ]:
# Accuracy
sns.set(style="whitegrid")

groups = (
    "767 Bands",
    "3 Feature \n Bands",
    "Literature \n Intersection \n Bands",
    "Literature \n Bands",
)
accuracy_values = {"Model 1": [], "Model 2": [], "Model 3": []}

for i in range(0, len(file_array), 3):
    try:
        architecture1 = float(file_array[i].iloc[99]["accuracy"])
        architecture2 = float(file_array[i + 1].iloc[99]["accuracy"])
        architecture3 = float(file_array[i + 2].iloc[99]["accuracy"])

        accuracy_values["Model 1"].append(architecture1)
        accuracy_values["Model 2"].append(architecture2)
        accuracy_values["Model 3"].append(architecture3)

    except IndexError as e:
        print(f"IndexError at index {i}: {e}")
    except KeyError as e:
        print(f"KeyError at index {i}: {e}")

x = np.arange(len(groups))
width = 0.25
multiplier = 0

fig, ax = plt.subplots()

for attribute, measurement in accuracy_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1


ax.set_ylabel("Training Accuracy")
ax.set_xticks(x + width / 2)
ax.set_xticklabels(groups)
ax.set_ylim(0, 1)

# Place the legend outside the plot
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)

sns.despine()

plt.tight_layout()
plt.savefig("bar_graph_acc.png")
plt.show()

# In[ ]:
# Loss
sns.set(style="whitegrid")

groups = (
    "767 Bands",
    "3 Feature \n Bands",
    "Literature \n Intersection \n Bands",
    "Literature \n Bands",
)
loss_values = {"Model 1": [], "Model 2": [], "Model 3": []}

for i in range(0, len(file_array), 3):
    try:
        architecture1 = float(file_array[i].iloc[99]["loss"])
        architecture2 = float(file_array[i + 1].iloc[99]["loss"])
        architecture3 = float(file_array[i + 2].iloc[99]["loss"])

        loss_values["Model 1"].append(architecture1)
        loss_values["Model 2"].append(architecture2)
        loss_values["Model 3"].append(architecture3)

    except IndexError as e:
        print(f"IndexError at index {i}: {e}")
    except KeyError as e:
        print(f"KeyError at index {i}: {e}")

x = np.arange(len(groups))
width = 0.25
multiplier = 0

fig, ax = plt.subplots()

for attribute, measurement in loss_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

ax.set_ylabel("Training Loss")
ax.set_xticks(x + width / 2)
ax.set_xticklabels(groups)
ax.legend()
ax.set_ylim(0, 1)

# Place the legend outside the plot
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
sns.despine()

plt.tight_layout()
plt.savefig("bar_graph_loss_acc.png")
plt.show()

# In[ ]:
# time

sns.set(style="whitegrid")

groups = (
    "767 Bands",
    "3 Feature \n Bands",
    "Literature \n Intersection \n Bands",
    "Literature \n Bands",
)
time_values = {"Model 1": [], "Model 2": [], "Model 3": []}

for i in range(0, len(file_array), 3):
    try:
        architecture1 = float(file_array[i].iloc[-1]["accuracy"])
        architecture2 = float(file_array[i + 1].iloc[-1]["accuracy"])
        architecture3 = float(file_array[i + 2].iloc[-1]["accuracy"])

        time_values["Model 1"].append(architecture1)
        time_values["Model 2"].append(architecture2)
        time_values["Model 3"].append(architecture3)

    except IndexError as e:
        print(f"IndexError at index {i}: {e}")
    except KeyError as e:
        print(f"KeyError at index {i}: {e}")

x = np.arange(len(groups))
width = 0.25
multiplier = 0

fig, ax = plt.subplots()

for attribute, measurement in time_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

ax.set_ylabel("Time (seconds)", fontsize=12)
# ax.set_title('Training Time', fontsize=14, fontweight='bold')
ax.set_xticks(x + width / 2, groups)
ax.legend(fontsize=10)
ax.set_ylim(10, 55)
# Place the legend outside the plot
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)

sns.despine()

plt.tight_layout()
plt.savefig("bar_graph_time.png")

plt.show()

# In[ ]:
# validation loss graph

sns.set(style="whitegrid")

groups = (
    "767 Bands",
    "3 Feature \n Bands",
    "Literature \n Intersection \n Bands",
    "Literature \n Bands",
)
accuracy_values = {"Model 1": [], "Model 2": [], "Model 3": []}

for i in range(0, len(file_array), 3):
    architecture1 = file_array[i].iloc[99]["val_loss"]
    architecture2 = file_array[i + 1].iloc[99]["val_loss"]
    architecture3 = file_array[i + 2].iloc[99]["val_loss"]

    accuracy_values["Model 1"].append(architecture1)
    accuracy_values["Model 2"].append(architecture2)
    accuracy_values["Model 3"].append(architecture3)

x = np.arange(len(groups))
width = 0.25
multiplier = 0

fig, ax = plt.subplots()

for attribute, measurement in accuracy_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1


ax.set_ylabel("Validation Loss")
# ax.set_title('Validation Loss')
ax.set_xticks(x + width, groups)
ax.legend()
ax.set_ylim(0, 1)

# Place the legend outside the plot
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
sns.despine()
plt.tight_layout()
plt.savefig("bar_graph_val_loss.png")

plt.show()

# In[ ]:
# validation accuracy graph
sns.set(style="whitegrid")

groups = (
    "767 Bands",
    "3 Feature \n Bands",
    "Literature \n Intersection \n Bands",
    "Literature \n Bands",
)
accuracy_values = {"Model 1": [], "Model 2": [], "Model 3": []}

for i in range(0, len(file_array), 3):
    architecture1 = file_array[i].iloc[99]["val_accuracy"]
    architecture2 = file_array[i + 1].iloc[99]["val_accuracy"]
    architecture3 = file_array[i + 2].iloc[99]["val_accuracy"]

    accuracy_values["Model 1"].append(architecture1)
    accuracy_values["Model 2"].append(architecture2)
    accuracy_values["Model 3"].append(architecture3)

x = np.arange(len(groups))
width = 0.25
multiplier = 0

fig, ax = plt.subplots()

for attribute, measurement in accuracy_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1


ax.set_ylabel("Validation Accuracy")
# ax.set_title('Validation Accuracy')
ax.set_xticks(x + width, groups)
ax.legend()
ax.set_ylim(0, 1)

# Place the legend outside the plot
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
sns.despine()
plt.tight_layout()
plt.savefig("bar_graph_val_acc.png")

plt.show()

# In[ ]:
# Test Accuracy
sns.set(style="whitegrid")

groups = (
    "767 Bands",
    "3 Feature \n Bands",
    "Literature \n Intersection \n Bands",
    "Literature \n Bands",
)
loss_values = {"Model 1": [], "Model 2": [], "Model 3": []}

for i in range(0, len(file_array), 3):
    try:
        architecture1 = float(file_array[i].iloc[-5]["accuracy"])
        architecture2 = float(file_array[i + 1].iloc[-5]["accuracy"])
        architecture3 = float(file_array[i + 2].iloc[-5]["accuracy"])

        loss_values["Model 1"].append(architecture1)
        loss_values["Model 2"].append(architecture2)
        loss_values["Model 3"].append(architecture3)

    except IndexError as e:
        print(f"IndexError at index {i}: {e}")
    except KeyError as e:
        print(f"KeyError at index {i}: {e}")

x = np.arange(len(groups))
width = 0.25
multiplier = 0

fig, ax = plt.subplots()

for attribute, measurement in loss_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

ax.set_ylabel("Test Accuracy")
ax.set_xticks(x + width / 2)
ax.set_xticklabels(groups)
ax.legend()
ax.set_ylim(0, 1)

# Place the legend outside the plot
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
sns.despine()

plt.tight_layout()
plt.savefig("bar_graph_test_acc.png")
plt.show()

# In[ ]:
# Test Loss
sns.set(style="whitegrid")

groups = (
    "767 Bands",
    "3 Feature \n Bands",
    "Literature \n Intersection \n Bands",
    "Literature \n Bands",
)
loss_values = {"Model 1": [], "Model 2": [], "Model 3": []}

for i in range(0, len(file_array), 3):
    try:
        architecture1 = float(file_array[i].iloc[-6]["accuracy"])
        architecture2 = float(file_array[i + 1].iloc[-6]["accuracy"])
        architecture3 = float(file_array[i + 2].iloc[-6]["accuracy"])

        loss_values["Model 1"].append(architecture1)
        loss_values["Model 2"].append(architecture2)
        loss_values["Model 3"].append(architecture3)

    except IndexError as e:
        print(f"IndexError at index {i}: {e}")
    except KeyError as e:
        print(f"KeyError at index {i}: {e}")

x = np.arange(len(groups))
width = 0.25
multiplier = 0

fig, ax = plt.subplots()

for attribute, measurement in loss_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

ax.set_ylabel("Test Loss")
ax.set_xticks(x + width / 2)
ax.set_xticklabels(groups)
ax.legend()
ax.set_ylim(0, 1)

# Place the legend outside the plot
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
sns.despine()

plt.tight_layout()
plt.savefig("bar_graph_test_loss.png")
plt.show()

# In[ ]:
# a767_band_75
# a3_feat_123
# alit_inter_300
# alit_band_300

sns.set(style="ticks", font_scale=1.2)
sns.color_palette("Set1")

# Extract accuracy data for the first 99 epochs
accuracy1 = a767_band_300["accuracy"].iloc[:99]
accuracy2 = a3_feat_300["accuracy"].iloc[:99]
accuracy3 = alit_inter_120["accuracy"].iloc[:99]
accuracy4 = alit_band_300["accuracy"].iloc[:99]

# Generate epoch range
epochs = range(1, len(accuracy1) + 1)

# Plot the accuracy vs epoch graph for a767_band_75
plt.figure(figsize=(8, 6))
plt.plot(epochs, accuracy1, label="767 Bands, Model 1")
plt.plot(epochs, accuracy2, label="3 Feature Bands, Model 1")
plt.plot(epochs, accuracy3, label="Literature Intersection Bands, Model 2")
plt.plot(epochs, accuracy4, label="Literature Bands, Model 1")

# plt.title('Accuracy', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.ylim(0.6, 1)

sns.despine()
plt.tight_layout()
plt.savefig("acc_epoch_graph.png")
plt.show()


# Loss vs Epoch Graph

loss1 = a767_band_300["loss"].iloc[:99]
loss2 = a3_feat_300["loss"].iloc[:99]
loss3 = alit_inter_120["loss"].iloc[:99]
loss4 = alit_band_300["loss"].iloc[:99]

epochs = range(1, len(loss1) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, loss1, label="767 Bands, Model 1")
plt.plot(epochs, loss2, label="3 Feature Bands, Model 1")
plt.plot(epochs, loss3, label="Literature Intersection Bands, Model 2")
plt.plot(epochs, loss4, label="Literature Bands, Model 1")


# plt.title('Loss', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.ylim(0.3, 1)

sns.despine()
plt.tight_layout()
plt.savefig("loss_epoch_graph.png")
plt.show()


# Val Accuracy vs Epoch

valacc1 = a767_band_300["val_accuracy"]
valacc2 = a3_feat_300["val_accuracy"]
valacc3 = alit_inter_120["val_accuracy"]
valacc4 = alit_band_300["val_accuracy"]

epochs = range(1, len(valacc1) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, valacc1, label="767 Bands, Model 1")
plt.plot(epochs, valacc2, label="3 Feature Bands, Model 1")
plt.plot(epochs, valacc3, label="Literature Intersection Bands, Model 2")
plt.plot(epochs, valacc4, label="Literature Bands, Model 1")


# plt.title('Validation Accuracy', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.ylim(0.6, 1)

sns.despine()
plt.tight_layout()
plt.savefig("val_acc_epoch_graph.png")
plt.show()


# Val Loss vs Epoch

valloss1 = a767_band_300["val_loss"]
valloss2 = a3_feat_300["val_loss"]
valloss3 = alit_inter_120["val_loss"]
valloss4 = alit_band_300["val_loss"]

epochs = range(1, len(valloss1) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, valloss1, label="767 Bands, Model 1")
plt.plot(epochs, valloss2, label="3 Feature Bands, Model 1")
plt.plot(epochs, valloss3, label="Literature Intersection Bands, Model 2")
plt.plot(epochs, valloss4, label="Literature Bands, Model 1")


# plt.title('Validation Loss', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.ylim(0.3, 1)

sns.despine()
plt.tight_layout()
plt.savefig("val_loss_epoch_graph.png")
plt.show()

# In[ ]:

sns.set(style="ticks", font_scale=1.2)
sns.color_palette("Set1")
fig, axes = plt.subplots(figsize=(8, 6), dpi=600)


# Model 1 Accuracy Graph 1
accuracy767_300 = a767_band_300["loss"].iloc[:99]
accuracy767_120 = a767_band_120["loss"].iloc[:99]
accuracy767_75 = a767_band_75["loss"].iloc[:99]
accuracy3_300 = a3_feat_300["loss"].iloc[:99]
accuracy3_120 = a3_feat_120["loss"].iloc[:99]
accuracy3_75 = a3_feat_75["loss"].iloc[:99]
accuracylit_300 = alit_band_300["loss"].iloc[:99]
accuracylit_120 = alit_band_120["loss"].iloc[:99]
accuracylit_75 = alit_band_75["loss"].iloc[:99]
accuracyinter_300 = alit_inter_300["loss"].iloc[:99]
accuracyinter_120 = alit_inter_120["loss"].iloc[:99]
accuracyinter_75 = alit_inter_75["loss"].iloc[:99]

epochs = range(1, len(accuracy767_300) + 1)

plt.plot(epochs, accuracy767_300, label="767 Bands, Model 1")
plt.plot(epochs, accuracy3_300, label="3 Feature, Model 1")
plt.plot(epochs, accuracylit_300, label="Literature Bands, Model 1")
plt.plot(epochs, accuracyinter_300, label="Literature Intersection Bands, Model 1")


# plt.title('Accuracy', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.ylim(0.3, 1)

sns.despine()
plt.tight_layout()
plt.savefig("loss_epoch_graph_model_1.png")
plt.show()

##new graph
sns.set(style="ticks", font_scale=1.2)
sns.color_palette("Set1")
fig, axes = plt.subplots(figsize=(8, 6), dpi=600)

plt.plot(epochs, accuracy767_120, label="767 Bands, Model 2")
plt.plot(epochs, accuracy3_120, label="3 Feature, Model 2")
plt.plot(epochs, accuracylit_120, label="Literature Bands, Model 2")
plt.plot(epochs, accuracyinter_120, label="Literature Intersection Bands, Model 2")


# plt.title('Accuracy', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.ylim(0.3, 1)

sns.despine()
plt.tight_layout()
plt.savefig("loss_epoch_graph_model_2.png")
plt.show()

##new graph
sns.set(style="ticks", font_scale=1.2)
sns.color_palette("Set1")
fig, axes = plt.subplots(figsize=(8, 6), dpi=600)

plt.plot(epochs, accuracy767_75, label="767 Bands, Model 3")
plt.plot(epochs, accuracy3_75, label="3 Feature, Model 3")
plt.plot(epochs, accuracylit_75, label="Literature Bands, Model 3")
plt.plot(epochs, accuracyinter_75, label="Literature Intersection Bands, Model 3")


# plt.title('Accuracy', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.ylim(0.3, 1)

sns.despine()
plt.tight_layout()
plt.savefig("loss_epoch_graph_model_3.png")
plt.show()

# In[ ]:

sns.set(style="ticks", font_scale=1.2)
sns.color_palette("Set1")
fig, axes = plt.subplots(figsize=(8, 6), dpi=600)

# Model 1 Val Accuracy Graph 1
accuracy767_300 = a767_band_300["val_accuracy"]
accuracy767_120 = a767_band_120["val_accuracy"]
accuracy767_75 = a767_band_75["val_accuracy"]
accuracy3_300 = a3_feat_300["val_accuracy"]
accuracy3_120 = a3_feat_120["val_accuracy"]
accuracy3_75 = a3_feat_75["val_accuracy"]
accuracylit_300 = alit_band_300["val_accuracy"]
accuracylit_120 = alit_band_120["val_accuracy"]
accuracylit_75 = alit_band_75["val_accuracy"]
accuracyinter_300 = alit_inter_300["val_accuracy"]
accuracyinter_120 = alit_inter_120["val_accuracy"]
accuracyinter_75 = alit_inter_75["val_accuracy"]

epochs = range(1, len(accuracy767_300) + 1)

plt.plot(epochs, accuracy767_300, label="767 Bands, Model 1")
plt.plot(epochs, accuracy3_300, label="3 Feature, Model 1")
plt.plot(epochs, accuracylit_300, label="Literature Bands, Model 1")
plt.plot(epochs, accuracyinter_300, label="Literature Intersection Bands, Model 1")


# plt.title('Accuracy', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.ylim(0.6, 1)

sns.despine()
plt.tight_layout()
plt.savefig("val_acc_epoch_graph_model_1.png")
plt.show()

##new graph
sns.set(style="ticks", font_scale=1.2)
sns.color_palette("Set1")
fig, axes = plt.subplots(figsize=(8, 6), dpi=600)

plt.plot(epochs, accuracy767_120, label="767 Bands, Model 2")
plt.plot(epochs, accuracy3_120, label="3 Feature, Model 2")
plt.plot(epochs, accuracylit_120, label="Literature Bands, Model 2")
plt.plot(epochs, accuracyinter_120, label="Literature Intersection Bands, Model 2")


# plt.title('Accuracy', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.ylim(0.6, 1)

sns.despine()
plt.tight_layout()
plt.savefig("val_acc_epoch_graph_model_2.png")
plt.show()

##new graph
sns.set(style="ticks", font_scale=1.2)
sns.color_palette("Set1")
fig, axes = plt.subplots(figsize=(8, 6), dpi=600)

plt.plot(epochs, accuracy767_75, label="767 Bands, Model 3")
plt.plot(epochs, accuracy3_75, label="3 Feature, Model 3")
plt.plot(epochs, accuracylit_75, label="Literature Bands, Model 3")
plt.plot(epochs, accuracyinter_75, label="Literature Intersection Bands, Model 3")


# plt.title('Accuracy', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.ylim(0.6, 1)

sns.despine()
plt.tight_layout()
plt.savefig("val_acc_epoch_graph_model_3.png")
plt.show()

# In[ ]:

sns.set(style="ticks", font_scale=1.2)
sns.color_palette("Set1")
fig, axes = plt.subplots(figsize=(8, 6), dpi=600)

# Model 1 Val Accuracy Graph 1
accuracy767_300 = a767_band_300["val_loss"]
accuracy767_120 = a767_band_120["val_loss"]
accuracy767_75 = a767_band_75["val_loss"]
accuracy3_300 = a3_feat_300["val_loss"]
accuracy3_120 = a3_feat_120["val_loss"]
accuracy3_75 = a3_feat_75["val_loss"]
accuracylit_300 = alit_band_300["val_loss"]
accuracylit_120 = alit_band_120["val_loss"]
accuracylit_75 = alit_band_75["val_loss"]
accuracyinter_300 = alit_inter_300["val_loss"]
accuracyinter_120 = alit_inter_120["val_loss"]
accuracyinter_75 = alit_inter_75["val_loss"]

epochs = range(1, len(accuracy767_300) + 1)

plt.plot(epochs, accuracy767_300, label="767 Bands, Model 1")
plt.plot(epochs, accuracy3_300, label="3 Feature, Model 1")
plt.plot(epochs, accuracylit_300, label="Literature Bands, Model 1")
plt.plot(epochs, accuracyinter_300, label="Literature Intersection Bands, Model 1")


# plt.title('Accuracy', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.ylim(0.3, 1)

sns.despine()
plt.tight_layout()
plt.savefig("val_loss_epoch_graph_model_1.png")
plt.show()

##new graph
sns.set(style="ticks", font_scale=1.2)
sns.color_palette("Set1")
fig, axes = plt.subplots(figsize=(8, 6), dpi=600)

plt.plot(epochs, accuracy767_120, label="767 Bands, Model 2")
plt.plot(epochs, accuracy3_120, label="3 Feature, Model 2")
plt.plot(epochs, accuracylit_120, label="Literature Bands, Model 2")
plt.plot(epochs, accuracyinter_120, label="Literature Intersection Bands, Model 2")


# plt.title('Accuracy', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.ylim(0.3, 1)

sns.despine()
plt.tight_layout()
plt.savefig("val_loss_epoch_graph_model_2.png")
plt.show()

##new graph
sns.set(style="ticks", font_scale=1.2)
sns.color_palette("Set1")
fig, axes = plt.subplots(figsize=(8, 6), dpi=600)

plt.plot(epochs, accuracy767_75, label="767 Bands, Model 3")
plt.plot(epochs, accuracy3_75, label="3 Feature, Model 3")
plt.plot(epochs, accuracylit_75, label="Literature Bands, Model 3")
plt.plot(epochs, accuracyinter_75, label="Literature Intersection Bands, Model 3")


# plt.title('Accuracy', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.ylim(0.3, 1)

sns.despine()
plt.tight_layout()
plt.savefig("val_loss_epoch_graph_model_3.png")
plt.show()

# In[ ]:

sns.set(style="ticks", font_scale=1.2)
sns.color_palette("Set1")
fig, axes = plt.subplots(figsize=(8, 6), dpi=600)

# Model 1 Val Accuracy Graph 1
accuracy767_300 = a767_band_300["accuracy"]
accuracy767_120 = a767_band_120["accuracy"]
accuracy767_75 = a767_band_75["accuracy"]
accuracy3_300 = a3_feat_300["accuracy"]
accuracy3_120 = a3_feat_120["accuracy"]
accuracy3_75 = a3_feat_75["accuracy"]
accuracylit_300 = alit_band_300["accuracy"]
accuracylit_120 = alit_band_120["accuracy"]
accuracylit_75 = alit_band_75["accuracy"]
accuracyinter_300 = alit_inter_300["accuracy"]
accuracyinter_120 = alit_inter_120["accuracy"]
accuracyinter_75 = alit_inter_75["accuracy"]

epochs = range(1, len(accuracy767_300) + 1)

plt.plot(epochs, accuracy767_300, label="767 Bands, Model 1")
plt.plot(epochs, accuracy3_300, label="3 Feature, Model 1")
plt.plot(epochs, accuracylit_300, label="Literature Bands, Model 1")
plt.plot(epochs, accuracyinter_300, label="Literature Intersection Bands, Model 1")


# plt.title('Accuracy', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.ylim(0.6, 1)

sns.despine()
plt.tight_layout()
plt.savefig("acc_epoch_graph_model_1.png")
plt.show()

##new graph
sns.set(style="ticks", font_scale=1.2)
sns.color_palette("Set1")
fig, axes = plt.subplots(figsize=(8, 6), dpi=600)

plt.plot(epochs, accuracy767_120, label="767 Bands, Model 2")
plt.plot(epochs, accuracy3_120, label="3 Feature, Model 2")
plt.plot(epochs, accuracylit_120, label="Literature Bands, Model 2")
plt.plot(epochs, accuracyinter_120, label="Literature Intersection Bands, Model 2")


# plt.title('Accuracy', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.ylim(0.6, 1)

sns.despine()
plt.tight_layout()
plt.savefig("acc_epoch_graph_model_2.png")
plt.show()

##new graph
sns.set(style="ticks", font_scale=1.2)
sns.color_palette("Set1")
fig, axes = plt.subplots(figsize=(8, 6), dpi=600)

plt.plot(epochs, accuracy767_75, label="767 Bands, Model 3")
plt.plot(epochs, accuracy3_75, label="3 Feature, Model 3")
plt.plot(epochs, accuracylit_75, label="Literature Bands, Model 3")
plt.plot(epochs, accuracyinter_75, label="Literature Intersection Bands, Model 3")


# plt.title('Accuracy', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.ylim(0.6, 1)

sns.despine()
plt.tight_layout()
plt.savefig("acc_epoch_graph_model_3.png")
plt.show()
