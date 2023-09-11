#%%
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

log_dir = 'models\model4\log'
fig_save = 'models\model4\imgs\loss.png'
event_acc = event_accumulator.EventAccumulator(log_dir)
event_acc.Reload()
tags = event_acc.Tags()

loss_list = []
loss_events = event_acc.Scalars('loss')
for event in loss_events:
    print(f"Step: {event.step}, Value: {event.value}")
    loss_list.append(event.value)

plt.plot(loss_list)
plt.savefig(fig_save)
# %%
