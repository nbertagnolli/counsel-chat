# counsel-chat
This repository holds the code for working with data from counselchat.com.   The scarped data are from individiuals seeking assistance from licensed therapists and their associated responses. The goal is to provide a high quality open source dataset of quality counseling responses.

I've recently added the data to [HuggingFace](https://huggingface.co/datasets/nbertagnolli/counsel-chat) so getting the data should be as easy as:

```python
from datasets import load_dataset

dataset = load_dataset("nbertagnolli/counsel-chat")
```

There is a larger writeup available on [medium](https://medium.com/towards-data-science/counsel-chat-bootstrapping-high-quality-therapy-data-971b419f33da)

If you use this data in your work please cite the medium article.

```
@misc{bertagnolli2020counsel,
  title={Counsel chat: Bootstrapping high-quality therapy data},
  author={Bertagnolli, Nicolas},
  year={2020},
  publisher={Towards Data Science. https://towardsdatascience. com/counsel-chat~…}
}
```


