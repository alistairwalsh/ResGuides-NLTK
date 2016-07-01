
<br>
<img style="float:left" src="http://ipython.org/_static/IPy_header.png" />
<br>

# Session 1: python syntax, variables, functions, list and loops

<br>
Welcome to the *IPython Notebook*. Through this interface, you'll be learning a lot of things:

* How to use iPython

* Some simple commands and syntax for exploring vocab with programming, specifically, the python library nltk

* How to use lists and indexing with Python to explore vocab

* Defining variables and functions in python

* Learning about loops and automating commands

* Turning words into data: tokenizing with nltk

* How to use nltk to remove common words and aid meaningful textual analysis

* Web scraping with nltk and beautiful soup

You can head [here](https://github.com/resbaz/lessons/blob/master/nltk/README.md) for the fully articulated overview of the course, but we'll almost always stay within IPython. 
Remember, everything we cover here will remain available to you after ResBaz is over, including these Notebooks. It's all accessible at the [ResBaz GitHub](https://github.com/resbaz/lessons/tree/master/nltk).

**Where am I!?**

Fair question. While your arrival might have been disorientating, you are on the cloud, using a virtual machine. Your virtual machine has its own command line or terminal, if you're using a Mac or Linux operating systems. In the old days, it was the only user interface available on a Unix computer. Nowadays, we have graphical user interfaces (GUIs) in addition to command line interfaces. But there are things you can do from the command line that you can't do with a GUI, such as clone a repository, like you just did. 

This virutal machine has been setup to run an application. That application is the Jupyter Notebook, which is a spin-off of IPython, a tool for interactive computing on the web. Python is typically written as a standalone program on your computer. We run it off the cloud because of the amount of data we're using and also because it's cool. 

If you'd like to install Python yourself, the best way to do it is to download [Anaconda](https://www.continuum.io/downloads). It's free. 

The method for installing nltk is slightly different from what we are about to do on the cloud. Once you've installed the suite of tools that come with Anaconda (which includes the Jupyter Notebook), install NLTK from [here](http://www.nltk.org/install.html). Next, open the Jupyter Notebook on your machine and type "import nltk" into a cell (without the quotations). 

Please note that if you are using very large datasets it will be slow. You can get your own cloud environment as a research for free. Research Platforms can assist you with that process if you need it.

**Any questions before we begin?**

Lets take a tour of the Jupyter Notebook.

So, now that you've had a play with the Jupyter notebook, some words on its utility.

1. The main strength of IPython is that you can run bits of code individually, so you don't have to keep repeating things. For example, if you scroll up to the last function and replace the 50 with 2, you can re-run that code and get the new answer. 
2. IPython allows you to display images alongside code, and to save the input and output together.
3. IPython makes learning a bit easier, as mistakes are easier to find and do not break an entire workflow.

## Text as data

Programming languages like Python are great for processing data. In order to apply it to *text*, we need to think about our text as data.
This means being aware of how text is structured, what extra information might be encoded in it, and how to manage to give the best results. 

## What is the Natural Language Toolkit?

<br>
We'll be covering some of the theory behind corpus linguistics later on, but let's start by looking at some of the tasks NLTK can help you with. 

NLTK is a Python Library for working with written language data. It is free and extensively documented. Many areas we'll be covering are treated in more detail in the NLTK Book, available free online from [here](http://www.nltk.org/book/).

> Note: NLTK provides tools for tasks ranging from very simple (counting words in a text) to very complex (writing and training parsers, etc.). Many advanced tasks are beyond the scope of this course, but by the time we're done, you should understand Python and NLTK well enough to perform these tasks on your own!

We will start by importing NLTK, setting a path to NLTK resources, and downloading some additional stuff.


```python
from __future__ import print_function, division
# clear output from download
from IPython.display import display, clear_output
# import: all the nltk basics
import nltk
```

Oh, we've got to import some corpora used in the book as well...


```python
from nltk.book import *  
# asterisk means 'everything'
```

    *** Introductory Examples for the NLTK Book ***
    Loading text1, ..., text9 and sent1, ..., sent9
    Type the name of the text or sentence to view it.
    Type: 'texts()' or 'sents()' to list the materials.
    text1: Moby Dick by Herman Melville 1851
    text2: Sense and Sensibility by Jane Austen 1811
    text3: The Book of Genesis
    text4: Inaugural Address Corpus
    text5: Chat Corpus
    text6: Monty Python and the Holy Grail
    text7: Wall Street Journal
    text8: Personals Corpus
    text9: The Man Who Was Thursday by G . K . Chesterton 1908


Importing the book has assigned variable names to ten corpora. We can call these names easily: 


```python
text2
#text3
```




    <Text: Sense and Sensibility by Jane Austen 1811>



### Variables: why we use them

By assigning a variable you can save data to particular values. For example, Moby Dick is stored as Text1

Varibales are handy short cuts to our data and can be used in functions to determine the length of the data and so much more

Try assigning your own variable using this syntax: my_variable_name = 'the quality of mercy is not strained...'


```python
my_variable_name = 'the quality of mercy is not strained...'
```

Now "call" it. You can call a variable at anytime if you forget what data you saved to it, like we did above. Try using Tab complete.


```python
my_variable_name
```




    'the quality of mercy is not strained...'



### Python syntax
The syntax of the Python programming language is the set of rules that defines how a Python program will be written and interpreted (by both computers and by human readers).

Python was designed to be a highly readable language. It has a relatively uncluttered visual layout and uses English keywords frequently where other languages use punctuation. So it's really the perfect computer language for our purposes.

The syntax we'll use the most are two types of commands; one that look like this len() and anothers that look like this .count() 

Both need an object (text data in our case) to work on; for example, len(text1) or text1.count("Whale").

We are now going to play around with the first type of command.

### Exploring vocabulary

NLTK makes it really easy to get basic information about the size of a text and the complexity of its vocabulary.

**len(text1)** gives the number of symbols or 'tokens' in your text. This is the total number of words and items of punctuation.

**set(text2)** gives you a list of all the tokens in the text, without the duplicates.

Hence, **len(set(text3))** will give you the total number unique tokens. Remember this still includes punctuation. 

**sorted(text4)** places items in the list into alphabetical order, with punctuation symbols and capitalised words first.

*Please* note that all these commands use the same *syntax*; this is the first python syntax we'll learn.


```python
len(text3)
```




    44764




```python
len(set(text3))
```




    2789




```python
sorted(set(text3)) [25:35]
```




    [u'Achbor',
     u'Adah',
     u'Adam',
     u'Adbeel',
     u'Admah',
     u'Adullamite',
     u'After',
     u'Aholibamah',
     u'Ahuzzath',
     u'Ajah']



### Using iPython as a calculator

We can use the iPython environment as an overblown caluculator; try doing some basic mathematics with python. *Hint*: use * and / like your smartphone.


```python
3 * 4
```




    12



The expression that you just wrote above is the most basic programming instruction in the Python language. It includes values (the numbers) and operators (* + - etc...) and always evaluate/reduce down to one value. 

Operators are important and we can do more with them than multiply. We can ask IPython if something is equal to == or not equal to != and a number of others. Try it!


```python
3 == 4
```




    False




```python
3 != 4
```




    True



Handy! This introduces us to boolean operators. IPython is telling me whether the expressions I've type are True or False. More on operators later.

**Challenge!** 

We can investigate the *lexical richness* of a text. For example, by dividing the total number of words by the number of unique words, we can see the average number of times each word is used. 

For this challenge you will have to combine your knowledge of the syntax we've learnt so far and iPython's mathematical abilities.

Have a go at calculating the lexical richness of text3.


```python
len(text3)/len(set(text3))
```




    16



We can also count the number of times a word is used and calculate what percentage of the text it represents. But to do this we need to learn some new *syntax*. Methods that use dot notation only work with strings; on the other hand, len() and sorted() can work on other data types.


```python
text4.count("America")
```




    192



**Challenge!** 

How would you calculate the percentage of Text 4 that is taken up by the word "America"?


```python
100*(text4.count("America")/len(text4)) 
```




    0.13174597728754248



### Exploring text - concordances, similar contexts, dispersion

'Concordance' shows you a word in context and is useful if you want to be able to discuss the ways in which a word is used in a text. 
'Similar' will find words used in similar contexts; remember it is not looking for synonyms, 
although the results may include synonyms


```python
text1.concordance("monstrous")
```

    Displaying 11 of 11 matches:
    ong the former , one was of a most monstrous size . ... This came towards us , 
    ON OF THE PSALMS . " Touching that monstrous bulk of the whale or ork we have r
    ll over with a heathenish array of monstrous clubs and spears . Some were thick
    d as you gazed , and wondered what monstrous cannibal and savage could ever hav
    that has survived the flood ; most monstrous and most mountainous ! That Himmal
    they might scout at Moby Dick as a monstrous fable , or still worse and more de
    th of Radney .'" CHAPTER 55 Of the Monstrous Pictures of Whales . I shall ere l
    ing Scenes . In connexion with the monstrous pictures of whales , I am strongly
    ere to enter upon those still more monstrous stories of them which are to be fo
    ght have been rummaged out of this monstrous cabinet there is no telling . But 
    of Whale - Bones ; for Whales of a monstrous size are oftentimes cast up dead u



```python
text1.similar("monstrous")
```

    imperial subtly impalpable pitiable curious abundant perilous
    trustworthy untoward singular lamentable few determined maddens
    horrible tyrannical lazy mystifying christian exasperate



```python
text2.similar("monstrous")
```

    very exceedingly so heartily a great good amazingly as sweet
    remarkably extremely vast



```python
text2.common_contexts(["monstrous", "very"])  # this function takes two arguments
```

    a_pretty is_pretty a_lucky am_glad be_glad


We can also find words that typically occur together, which tend to be very specific to a text or genre of texts. We'll talk more about these features and how to use them later.


```python
text4.collocations()
```

    United States; fellow citizens; four years; years ago; Federal
    Government; General Government; American people; Vice President; Old
    World; Almighty God; Fellow citizens; Chief Magistrate; Chief Justice;
    God bless; every citizen; Indian tribes; public debt; one another;
    foreign nations; political parties



```python
text3.collocations()
```

    said unto; pray thee; thou shalt; thou hast; thy seed; years old;
    spake unto; thou art; LORD God; every living; God hath; begat sons;
    seven years; shalt thou; little ones; living creature; creeping thing;
    savoury meat; thirty years; every beast


**Challenge!**

1. Find the collocations in Gensis. 

2. Chose one of the creepier words to concordance. 

3. Investigate how the word is used. What words are used similarly? 

4. And what are the common contexts of these words? 

5. Report your findings to the person next to you. 

6. Do the same with the chat forum.

Python also lets you create graphs to display data.
To represent information about a text graphically, import the Python library *numpy*. We can then generate a dispersion plot that shows where given words occur in a text.


```python
import numpy
%matplotlib inline
text1.dispersion_plot(["whale"])
```


![png](output_50_0.png)



```python
text1.count('whale')
```




    906



**Challenge!**
<br>
Create a dispersion plot for the terms "citizens", "democracy", "freedom", "duties" and "America" in the innaugural address corpus.
What do you think it tells you? 


```python
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"]) # plot five words longitudinally
```


![png](output_53_0.png)



```python
text5.dispersion_plot([":)", ":("])
```


![png](output_54_0.png)


### Writing our own functions (yay!)

One thing that makes Python unique is that whitespace at the start of the line (use four spaces for consistency!) is meaningful. 
In many other languages, whitespace at the start of lines is simply a readability convention.


```python
# Fix this whitespace problem!

string = 'user'
if string == 'user':
print('Phew, fixed.')
```


      File "<ipython-input-20-252919a0783a>", line 5
        print('Phew, fixed.')
            ^
    IndentationError: expected an indented block



So, whitespace tells both Python and human readers where things start and stop.

### Defining a function

Here's an example:


```python
def welcomer(name):
    print('Welcome, %s!' % name)# here '%s' tells Python to expect a string and how many strings to expect.  
```

Here, the word 'name' is a placeholder. It stands in for any argument we might care to place in the brackets. The placeholder could be anything. It could be n or fsdlkfjs; it will still work. What matters for the program is that you use the same one consistently. On the other hand, what matters for us is that you use *descriptive* names so you remember what the code does! 

Notice that it doesn't do anything by itself. It needs to actually be *called*, and given some data:


```python
welcomer('Yuandra')
```

    Welcome, Yuandra!


Advantages of functions:
1. Save you typing
2. You can be sure you're doing exactly the same operation every time
<br>
> **Note** Learn to love tab-completion! Typing the first one or two letters of a command you've used previously then hitting tab 
will auto-complete that command, saving you typing (i.e. time and mistakes!). 

You may wish to repeat an operation multiple times looking at different texts or different terms within a text. Instead of re-entering the formula every time, you can assign a name and set of actions to a particular task.

Previously, we calculated the lexical diversity of a text. In NLTK, we can create a function called **lexical diversity** that runs a single line of code. We can then call this function to quickly determine the lexical density of a corpus or subcorpus.
<markdown cell>
**Challenge!**

Write a function to calculate the lexical diversity of a text; test it out on the books in the NLTK corpus


```python
def lexical_diversity(text):
    return len(text)/len(set(text))
```


```python
#After the function has been defined, we can run it:
lexical_diversity(text2)
```




    20



The parentheses are important here as they sepatate the the task, that is the work of the function, from the data that the function is to be performed on. 

The data in parentheses is called the argument of the function. When we use a function, we say that we 'call' it. 

Other functions that we've used already include len() and sorted() - these were predefined. *lexical_diversity()* is one we set up ourselves; note that it's conventional to put a set of parentheses after a function, to make it clear what we're talking about.

### Lists

Python treats a text as a long list of words. First, we'll make some lists of our own, to give you an idea of how a list behaves.


```python
sent1 = ['Call', 'me', 'Ishmael', '.']
# Note we use Square brackets here to define our list
```


```python
sent1
```




    ['Call', 'me', 'Ishmael', '.']




```python
len(sent1)
```




    4



The opening sentences of each of our texts have been pre-defined for you. You can inspect them by typing in 'sent2' etc.

You can add lists together, creating a new list containing all the items from both lists. You can do this by typing out the two lists or you can add two or more pre-defined lists. This is called concatenation.


```python
sent4 + sent1
```




    ['Fellow',
     '-',
     'Citizens',
     'of',
     'the',
     'Senate',
     'and',
     'of',
     'the',
     'House',
     'of',
     'Representatives',
     ':',
     'Call',
     'me',
     'Ishmael',
     '.']



We can also add an item to the end of a list by appending. When we append(), the list itself is updated. 


```python
sent1.append('Please')
sent1
```




    ['Call', 'me', 'Ishmael', '.', 'Please']



###  Indexing Lists

We can navigate this list with the help of indexes. Just as we can find out the number of times a word occurs in a text, we can also find where a word first occurs. We can navigate to different points in a text without restriction, so long as we can describe where we want to be.


```python
print(text4.index('awaken'))
```

    173


This works in reverse as well. We can ask Python to locate the 158th item in our list (note that we use square brackets here, not parentheses)


```python
print(text4[173])
```

    awaken


As well as pulling out individual items from a list, indexes can be used to pull out selections of text from a large corpus to inspect. We call this slicing.


```python
print(text5[16715:16735])
```

    [u'U86', u'thats', u'why', u'something', u'like', u'gamefly', u'is', u'so', u'good', u'because', u'you', u'can', u'actually', u'play', u'a', u'full', u'game', u'without', u'buying', u'it']


If we're asking for the beginning or end of a text, we can leave out the first or second number. For instance, [:5] will give us the first five items in a list while [8:] will give us all the elements from the eighth to the end. 


```python
print(text2[:10])print(text4[145700:])
```

    [u'[', u'Sense', u'and', u'Sensibility', u'by', u'Jane', u'Austen', u'1811', u']', u'CHAPTER']
    [u'upon', u'us', u',', u'we', u'carried', u'forth', u'that', u'great', u'gift', u'of', u'freedom', u'and', u'delivered', u'it', u'safely', u'to', u'future', u'generations', u'.', u'Thank', u'you', u'.', u'God', u'bless', u'you', u'.', u'And', u'God', u'bless', u'the', u'United', u'States', u'of', u'America', u'.']


To help you understand how indexes work, let's create one.

We start by defining the name of our index and then add the items. You probably won't do this in your own work, but you may want to manipulate an index in other ways. Pay attention to the quote marks and commas when you create your test sentence.


```python
sent = ['The', 'quick', 'brown', 'fox']
print(sent[0])
print(sent[2])
```

    The
    brown


Note that the first element in the list is zero. This is because we are telling Python to go zero steps forward in the list. If we use an index that is too large (that is, we ask for something that doesn't exist), we'll get an error.

We can modify elements in a list by assigning new data to one of its index values. We can also replace a slice with new material.


```python
sent[2] = 'furry'
sent[3] = 'child'
print(sent)
```

    ['The', 'quick', 'furry', 'child']


### Loops

We can use Python to automate tasks, such as performing a function on all items in a list. For instance, we could ask it to tell us the size of all the files in a directory. To do that, we'll have to teach the computer how to repeat things. We do this by creating something called a *loop*.

An example task that we might want to repeat is printing each character in a word on a line of its own. One way to do this would be to use a series of print statements:


```python
word = 'Python'
print(word[0])
print(word[1])
print(word[2])
print(word[3])
print(word[4])
print(word[5])
```

    P
    y
    t
    h
    o
    n


What are the cons of doing this do you think?


```python
word = 'Monty'
print(word[0])
print(word[1])
print(word[2])
print(word[3])
print(word[4])
print(word[5])
```

    M
    o
    n
    t
    y



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-38-6af80786acd1> in <module>()
          5 print(word[3])
          6 print(word[4])
    ----> 7 print(word[5])
    

    IndexError: string index out of range


There's a better way!


```python
word = 'Python'
for char in word:
    print(char)
```

    P
    y
    t
    h
    o
    n


This is call a 'for loop'. Try it with other words.

The general form of a loop is: 


```python
for variable in collection:
    do things with variable
```

We can call the loop variable whatever we want, but there must be a colon at the end of the line starting the loop, and we must indent anything we want to run inside the loop. This is used to signify when the loop ends, instead of using a symbol to end the loop.

This is a loop that repeatedly updates a variable:


```python
length = 0
for char in 'Python':
    length = length + 1
print('There are', length, 'letters in this word')
```

    There are 6 letters in this word


Let's go through this line by line. 

The variable is set to 0, so that python starts counting from 0. 

The second line opens a for loop that loops over the characters in Python. 

Now, the third line tells python to count by one, pretty obvious to us humans... Since there are six characters in 'Python', the statement on line 3 will be executed five times.

The first time around, length is zero (the value assigned to it on line 1). The statement adds 1 to the old value of length, producing 1, and updates length to refer to that new value. The next time around, char is 'y' and length is 1, so length is updated to be 2. After four more updates, length is 6.

Since there is nothing left in 'Python', the loop finishes and the print statement on line 4 tells us our final answer inbetween two strings.

Of course, we already know that we could use len() to find the length of a string, but it's just an example...


```python
len("Python")
```




    6



Now for another example using a list! Fruit salad, yummy, yummy.


```python
fruits = ['banana', 'apple', 'mango']
for fruit in fruits:        
    print('Current fruit :', fruit)
    print('Done!')
```

    Current fruit : banana
    Current fruit : apple
    Current fruit : mango
    Done!


**Challenge** 

Define a list called Library with the 9 NLTK books in it. Write a for loop that will run our lexical_diversity() function over each book and tell you its score.


```python
Library = [text1, text2, text3, text4, text5, text6, text7, text8, text9]
for book in Library:
    score = lexical_diversity(book)
    print(book, score)
```

    <Text: Moby Dick by Herman Melville 1851> 13
    <Text: Sense and Sensibility by Jane Austen 1811> 20
    <Text: The Book of Genesis> 16
    <Text: Inaugural Address Corpus> 14
    <Text: Chat Corpus> 7
    <Text: Monty Python and the Holy Grail> 7
    <Text: Wall Street Journal> 8
    <Text: Personals Corpus> 4
    <Text: The Man Who Was Thursday by G . K . Chesterton 1908> 10


###  Defining variables: revisted

In Python, we give the items we're working with names, a process called assignment. For instance, in the NLTK corpus, 'Sense and Sensibility' has been assigned the name 'text2', which is much easier to work with. 
We also assigend the name 'sent' to the sentence that we created in the previous exercise, so that we could then instruct Python to do various things with it. Assigning a variable in python looks like this:  
variable = expression   
You can call your variables (almost) anything you like, but it's a good idea to pick names that will be meaningful and easy to type. You can't use words that already have a meaning in Python, such as import, def, or not. If you try to use a word that is reserved, you'll get a syntax error.

**Challenge**

- Create a list called 'opening' that consists of the phrase "It was a dark and stormy night; the rain fell in torrents"
- Create a variable called 'clause' that contains the contents of 'opening', up to the semi-colon
- Create a variable called 'alphabetised' that contains the contents of 'clause' sorted alphabetically
- Print 'alphabetised' 


```python
opening = ['It', 'was', 'a', 'dark', 'and', 'stormy', 'night', ';', 'the', 'rain', 'fell', 'in', 'torrents']
clause = opening[0:7]
alphabetised = sorted(clause)
```

Note that assigning a variable just causes Python to remember that information without generating any output. 

If you want Python to show you the result, you have to ask for it (this is a good thing when you assign a variable to a very long list!).


```python
print(clause)
```

    ['It', 'was', 'a', 'dark', 'and', 'stormy', 'night']



```python
print(alphabetised)
```

    ['It', 'a', 'and', 'dark', 'night', 'stormy', 'was']


### Frequency distributions

We can use Python's ability to perform statistical analysis of data to do further exploration of vocabulary. For instance, we might want to be able to find the most common or least common words in a text. We'll start by looking at frequency distribution.


```python
from nltk.probability import FreqDist
from collections import Counter
fdist1 = FreqDist(text1)
```


```python
fdist1.most_common(50)
```


```python
fdist1['whale']
```




    906




```python
fdist1.plot(50, cumulative = True)
```


![png](output_123_0.png)


**Challenge!**

Let's compare the 15 most common tokens of the texts in the NLTK book. You could do this manually, but it will save you time and typing if you define a function and then loop it over the list, 'Library', that you created earlier.



```python
def common_words(text):
    return FreqDist(text).most_common(15)
common_words(text1)
```




    [(u',', 18713),
     (u'the', 13721),
     (u'.', 6862),
     (u'of', 6536),
     (u'and', 6024),
     (u'a', 4569),
     (u'to', 4542),
     (u';', 4072),
     (u'in', 3916),
     (u'that', 2982),
     (u"'", 2684),
     (u'-', 2552),
     (u'his', 2459),
     (u'it', 2209),
     (u'I', 2124)]




```python
for book in Library:
    words = common_words(book)
    print(book, words)
```

    <Text: Moby Dick by Herman Melville 1851> [(u',', 18713), (u'the', 13721), (u'.', 6862), (u'of', 6536), (u'and', 6024), (u'a', 4569), (u'to', 4542), (u';', 4072), (u'in', 3916), (u'that', 2982), (u"'", 2684), (u'-', 2552), (u'his', 2459), (u'it', 2209), (u'I', 2124)]
    <Text: Sense and Sensibility by Jane Austen 1811> [(u',', 9397), (u'to', 4063), (u'.', 3975), (u'the', 3861), (u'of', 3565), (u'and', 3350), (u'her', 2436), (u'a', 2043), (u'I', 2004), (u'in', 1904), (u'was', 1846), (u'it', 1568), (u'"', 1506), (u';', 1419), (u'she', 1333)]
    <Text: The Book of Genesis> [(u',', 3681), (u'and', 2428), (u'the', 2411), (u'of', 1358), (u'.', 1315), (u'And', 1250), (u'his', 651), (u'he', 648), (u'to', 611), (u';', 605), (u'unto', 590), (u'in', 588), (u'that', 509), (u'I', 484), (u'said', 476)]
    <Text: Inaugural Address Corpus> [(u'the', 9281), (u'of', 6970), (u',', 6840), (u'and', 4991), (u'.', 4676), (u'to', 4311), (u'in', 2527), (u'a', 2134), (u'our', 1905), (u'that', 1688), (u'be', 1460), (u'is', 1403), (u'we', 1141), (u'for', 1075), (u'by', 1036)]
    <Text: Chat Corpus> [(u'.', 1268), (u'JOIN', 1021), (u'PART', 1016), (u'?', 737), (u'lol', 704), (u'to', 658), (u'i', 648), (u'the', 646), (u'you', 635), (u',', 596), (u'I', 576), (u'a', 568), (u'hi', 546), (u'me', 415), (u'...', 412)]
    <Text: Monty Python and the Holy Grail> [(u':', 1197), (u'.', 816), (u'!', 801), (u',', 731), (u"'", 421), (u'[', 319), (u']', 312), (u'the', 299), (u'I', 255), (u'ARTHUR', 225), (u'?', 207), (u'you', 204), (u'a', 188), (u'of', 158), (u'--', 148)]
    <Text: Wall Street Journal> [(u',', 4885), (u'the', 4045), (u'.', 3828), (u'of', 2319), (u'to', 2164), (u'a', 1878), (u'in', 1572), (u'and', 1511), (u'*-1', 1123), (u'0', 1099), (u'*', 965), (u"'s", 864), (u'for', 817), (u'that', 807), (u'*T*-1', 806)]
    <Text: Personals Corpus> [(u',', 539), (u'.', 353), (u'/', 110), (u'for', 99), (u'to', 74), (u'and', 74), (u'lady', 68), (u'-', 66), (u'seeks', 60), (u'a', 52), (u'with', 44), (u'S', 36), (u'ship', 33), (u'&', 30), (u'relationship', 29)]
    <Text: The Man Who Was Thursday by G . K . Chesterton 1908> [(u',', 3488), (u'the', 3291), (u'.', 2717), (u'a', 1713), (u'of', 1710), (u'and', 1568), (u'"', 1336), (u'to', 1045), (u'in', 888), (u'I', 885), (u'he', 858), (u'that', 841), (u'his', 765), (u'was', 716), (u'you', 580)]


## Exploring Vocab continued

As well as counting individual words, we can count other features of vocabulary, such as how often words of different lengths occur. We do this by putting together a number of the commands we've already learned.

We could start like this: 

     [len(word) for word in text1]

... but this would print the length of every word in the whole book, so let's skip that bit!


```python
fdist2= FreqDist(len(word) for word in text1)
```


```python
fdist2.max()
```




    3




```python
fdist2.freq(3)
```




    0.19255882431878046



These last two commands tell us that the most common word length is 3, and that these 3 letter words account for about 20% of the book. 
We can see this just by visually inspecting the list produced by *fdist2.most_common()*, but if this list were too long to inspect readily, or we didn't want to print it, there are other ways to explore it.  

There are a number of functions defined for NLTK's frequency distributions:

 | Function | Purpose  |
 |--------------|------------|
 | fdist = FreqDist(samples) | create a frequency distribution containing the given samples |
 | fdist[sample] += 1 | increment the count for this sample |
 | fdist['monstrous']  | count of the number of times a given sample occurred |
 | fdist.freq('monstrous') | frequency of a given sample |
 | fdist.N()  |  total number of samples |
 | fdist.most_common(n)   |  the n most common samples and their frequencies |
 | for sample in fdist:   |  iterate over the items in fdist, when in the loop, we refer to each item as sample |
 | fdist.max() | sample with the greatest count |
 | fdist.tabulate()   |  tabulate the frequency distribution |
 | fdist.plot()  |   graphical plot of the frequency distribution |
 | fdist.plot(cumulative=True) | cumulative plot of the frequency distribution |
 | fdist1 < fdist2 | test if samples in fdist1 occur less frequently than in fdist2 |

It is possible to select the longest words in a text, which may tell you something about its vocabulary and style


```python
vocab = set(text4)
long_words = [word for word in vocab if len(word) > 15]
sorted(long_words)
```




    [u'RESPONSIBILITIES',
     u'antiphilosophists',
     u'constitutionally',
     u'contradistinction',
     u'discountenancing',
     u'disqualification',
     u'enthusiastically',
     u'instrumentalities',
     u'internationality',
     u'irresponsibility',
     u'misappropriation',
     u'misrepresentation',
     u'misunderstanding',
     u'responsibilities',
     u'sentimentalizing',
     u'transcontinental',
     u'uncharitableness',
     u'unconstitutional']



We can also use numerical operators to refine the types of searches we ask Python to run. We can use the following relational operators:


### Common relationals
 |  Relational | Meaning |
 |--------------:|:------------|
 | <    |  less than |
 | <=   |   less than or equal to |
 | ==  |    equal to (note this is two "=" signs, not one) |
 | !=   |   not equal to |
 | \>   |   greater than |
 | \>= |   greater than or equal to |

**Challenge!**

Using one of the pre-defined sentences in the NLTK corpus, use the relational operators above to find:

- Words longer than four characters
- Words of four or more characters
- Words of exactly four characters


```python
longer = [word for word in sent2 if len(word) > 4]
more = [word for word in sent2 if len(word) >= 4]
exact = [word for word in sent2 if len(word) == 4]
print(longer, more, exact)
```

    ['family', 'Dashwood', 'settled', 'Sussex'] ['family', 'Dashwood', 'long', 'been', 'settled', 'Sussex'] ['long', 'been']



```python
for word in sent2:
    if len(word) > 4:
        print(word)
```

    family
    Dashwood
    settled
    Sussex


We can fine-tune our selection even further by adding other conditions. For instance, we might want to find long words that occur frequently (or rarely).  

**Challenge!**

Can you find all the words in a text that are more than seven letters long and occur more than seven times?


```python
fdist5 = FreqDist(text5)
sorted(word for word in set(text5) if len(word) > 7 and fdist5[word] > 7)
```




    [u'#14-19teens',
     u'#talkcity_adults',
     u'((((((((((',
     u'........',
     u'Question',
     u'actually',
     u'anything',
     u'computer',
     u'cute.-ass',
     u'everyone',
     u'football',
     u'innocent',
     u'listening',
     u'remember',
     u'seriously',
     u'something',
     u'together',
     u'tomorrow',
     u'watching']



### Common operators

 | Operator  | Purpose  |
 |--------------|------------|
 | s.startswith(t) | test if s starts with t |
 | s.endswith(t)  |  test if s ends with t | 
 | t in s         |  test if t is a substring of s | 
 | s.islower()    |  test if s contains cased characters and all are lowercase | 
 | s.isupper()    |  test if s contains cased characters and all are uppercase | 
 | s.isalpha()    |  test if s is non-empty and all characters in s are alphabetic | 
 | s.isalnum()    |  test if s is non-empty and all characters in s are alphanumeric | 
 | s.isdigit()    |  test if s is non-empty and all characters in s are digits | 
 | s.istitle()    |  test if s contains cased characters and is titlecased (i.e. all words in s have initial capitals) | 


```python
sorted(w for w in set(text1) if w.endswith('ableness'))
```




    [u'comfortableness',
     u'honourableness',
     u'immutableness',
     u'indispensableness',
     u'indomitableness',
     u'intolerableness',
     u'palpableness',
     u'reasonableness',
     u'uncomfortableness']




```python
sorted(n for n in sent7 if n.isdigit())
```




    ['29', '61']



**Bonus!**

You'll remember right at the beginning we started looking at the size of the vocabulary of a text, but there were two problems with the results we got from using:

     len(set(text1)).

This count includes items of punctuation and treats capitalised and non-capitalised words as different things (*This* vs *this*). We can now fix these problems. We start by getting rid of capitalised words, then we get rid of the punctuation and numbers.


```python
len(set(text1))
```




    19317




```python
len(set(word.lower() for word in text1))
```




    17231




```python
len(set(word.lower() for word in text1 if word.isalpha()))
```




    16948



<img style="float:left" src="http://images.catsolonline.com/cache/custom/CEN/CE651-250x250.jpg" />
<br>
