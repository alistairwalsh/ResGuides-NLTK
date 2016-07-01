
<br>
<img style="float:left" src="http://ipython.org/_static/IPy_header.png" />
<br>

# Session 2: Loading text, tokenisation, tagging, dictionaries and ngrams


```python
from __future__ import print_function, division

import sys
import nltk
from IPython.display import display, clear_output
sys.path.append("/usr/lib/python2.7/site-packages/")
%matplotlib inline
```


```python
from nltk.book import *
```

**Welcome back!**

So, what did we learn yesterday? A brief recap:

* The **IPython** Notebook
* **Python**: syntax, variables, functions, etc.

Today's focus will be on **developing more advanced NLTK skills** and using these skills to **investigate our own data**. 

*Any questions or anything before we dive in?*

## Dirty data

Now that we're going beyond nltk example data, we're bond to run into dirty data.

A common part of corpus building is corpus cleaning. Reasons for cleaning include:

1. Not break the code with unexpected input
2. Ensure that searches match as many examples as possible
3. Increasing readability, the accuracy of taggers, stemmers, parsers, etc.

The level of kind of cleaning depends on your data, the aims of your project and where you are in your research. In the case of very clean data (lucky you!), there may be little that needs to be done. With messy data, you may need to go as far as to correct variant spellings (online conversation, very old books).

If you need help with data cleaning, we offer trainings in [OpenRefine](https://github.com/yuandra/2016-02-01-data-acquisition-cleaning/blob/gh-pages/open-refine-01-intro.md)

### Discussion

*What are the characteristics of clean and messy data? Any personal experiences? Discuss with your neighbours.* 

It will be important to bear these characteristics in mind once you start building your own datasets and corpora. 

## Uploading text files

First of all, let's load in our text.

Google the Gutenberg Project and download a book as a plain text file. 

I chose [A Modest Proposal](https://www.gutenberg.org/ebooks/1080)

We can also look at file contents within the IPython Notebook itself:


```python
import os
```


```python
# import tokenizers
from nltk import word_tokenize
from nltk.text import Text
```


```python
text_path = '/home/researcher/modest_proposal.txt'
```


```python
file = open(os.path.join(text_path), "r", encoding='UTF-8')
text = file.read()
print(text)
```

    ﻿The Project Gutenberg EBook of A Modest Proposal, by Jonathan Swift
    
    This eBook is for the use of anyone anywhere at no cost and with
    almost no restrictions whatsoever.  You may copy it, give it away or
    re-use it under the terms of the Project Gutenberg License included
    with this eBook or online at www.gutenberg.org
    
    
    Title: A Modest Proposal
           For preventing the children of poor people in Ireland,
           from being a burden on their parents or country, and for
           making them beneficial to the publick - 1729
    
    Author: Jonathan Swift
    
    Posting Date: July 27, 2008 [EBook #1080]
    Release Date: October 1997
    
    Language: English
    
    
    *** START OF THIS PROJECT GUTENBERG EBOOK A MODEST PROPOSAL ***
    
    
    
    
    Produced by An Anonymous Volunteer
    
    
    
    
    
    A MODEST PROPOSAL
    
    For preventing the children of poor people in Ireland, from being a
    burden on their parents or country, and for making them beneficial to
    the publick.
    
    by Dr. Jonathan Swift
    
    
    1729
    
    
    
    It is a melancholy object to those, who walk through this great town,
    or travel in the country, when they see the streets, the roads and
    cabbin-doors crowded with beggars of the female sex, followed by three,
    four, or six children, all in rags, and importuning every passenger for
    an alms. These mothers instead of being able to work for their honest
    livelihood, are forced to employ all their time in stroling to beg
    sustenance for their helpless infants who, as they grow up, either turn
    thieves for want of work, or leave their dear native country, to fight
    for the Pretender in Spain, or sell themselves to the Barbadoes.
    
    I think it is agreed by all parties, that this prodigious number of
    children in the arms, or on the backs, or at the heels of their mothers,
    and frequently of their fathers, is in the present deplorable state of
    the kingdom, a very great additional grievance; and therefore whoever
    could find out a fair, cheap and easy method of making these children
    sound and useful members of the common-wealth, would deserve so well of
    the publick, as to have his statue set up for a preserver of the nation.
    
    But my intention is very far from being confined to provide only for the
    children of professed beggars: it is of a much greater extent, and shall
    take in the whole number of infants at a certain age, who are born of
    parents in effect as little able to support them, as those who demand
    our charity in the streets.
    
    As to my own part, having turned my thoughts for many years, upon this
    important subject, and maturely weighed the several schemes of
    our projectors, I have always found them grossly mistaken in their
    computation. It is true, a child just dropt from its dam, may be
    supported by her milk, for a solar year, with little other nourishment:
    at most not above the value of two shillings, which the mother may
    certainly get, or the value in scraps, by her lawful occupation of
    begging; and it is exactly at one year old that I propose to provide for
    them in such a manner, as, instead of being a charge upon their parents,
    or the parish, or wanting food and raiment for the rest of their lives,
    they shall, on the contrary, contribute to the feeding, and partly to
    the cloathing of many thousands.
    
    There is likewise another great advantage in my scheme, that it will
    prevent those voluntary abortions, and that horrid practice of
    women murdering their bastard children, alas! too frequent among us,
    sacrificing the poor innocent babes, I doubt, more to avoid the expence
    than the shame, which would move tears and pity in the most savage and
    inhuman breast.
    
    The number of souls in this kingdom being usually reckoned one million
    and a half, of these I calculate there may be about two hundred thousand
    couple whose wives are breeders; from which number I subtract thirty
    thousand couple, who are able to maintain their own children, (although
    I apprehend there cannot be so many, under the present distresses of
    the kingdom) but this being granted, there will remain an hundred and
    seventy thousand breeders. I again subtract fifty thousand, for those
    women who miscarry, or whose children die by accident or disease within
    the year. There only remain an hundred and twenty thousand children of
    poor parents annually born. The question therefore is, How this number
    shall be reared, and provided for? which, as I have already said, under
    the present situation of affairs, is utterly impossible by all the
    methods hitherto proposed. For we can neither employ them in handicraft
    or agriculture; we neither build houses, (I mean in the country) nor
    cultivate land: they can very seldom pick up a livelihood by stealing
    till they arrive at six years old; except where they are of towardly
    parts, although I confess they learn the rudiments much earlier;
    during which time they can however be properly looked upon only as
    probationers: As I have been informed by a principal gentleman in the
    county of Cavan, who protested to me, that he never knew above one or
    two instances under the age of six, even in a part of the kingdom so
    renowned for the quickest proficiency in that art.
    
    I am assured by our merchants, that a boy or a girl before twelve years
    old, is no saleable commodity, and even when they come to this age, they
    will not yield above three pounds, or three pounds and half a crown
    at most, on the exchange; which cannot turn to account either to the
    parents or kingdom, the charge of nutriments and rags having been at
    least four times that value.
    
    I shall now therefore humbly propose my own thoughts, which I hope will
    not be liable to the least objection.
    
    I have been assured by a very knowing American of my acquaintance in
    London, that a young healthy child well nursed, is, at a year old, a
    most delicious nourishing and wholesome food, whether stewed, roasted,
    baked, or boiled; and I make no doubt that it will equally serve in a
    fricasie, or a ragoust.
    
    I do therefore humbly offer it to publick consideration, that of the
    hundred and twenty thousand children, already computed, twenty thousand
    may be reserved for breed, whereof only one fourth part to be males;
    which is more than we allow to sheep, black cattle, or swine, and my
    reason is, that these children are seldom the fruits of marriage, a
    circumstance not much regarded by our savages, therefore, one male will
    be sufficient to serve four females. That the remaining hundred thousand
    may, at a year old, be offered in sale to the persons of quality and
    fortune, through the kingdom, always advising the mother to let them
    suck plentifully in the last month, so as to render them plump, and fat
    for a good table. A child will make two dishes at an entertainment for
    friends, and when the family dines alone, the fore or hind quarter will
    make a reasonable dish, and seasoned with a little pepper or salt, will
    be very good boiled on the fourth day, especially in winter.
    
    I have reckoned upon a medium, that a child just born will weigh 12
    pounds, and in a solar year, if tolerably nursed, encreaseth to 28
    pounds.
    
    I grant this food will be somewhat dear, and therefore very proper for
    landlords, who, as they have already devoured most of the parents, seem
    to have the best title to the children.
    
    Infant's flesh will be in season throughout the year, but more plentiful
    in March, and a little before and after; for we are told by a grave
    author, an eminent French physician, that fish being a prolifick dyet,
    there are more children born in Roman Catholick countries about nine
    months after Lent, the markets will be more glutted than usual, because
    the number of Popish infants, is at least three to one in this kingdom,
    and therefore it will have one other collateral advantage, by lessening
    the number of Papists among us.
    
    I have already computed the charge of nursing a beggar's child (in which
    list I reckon all cottagers, labourers, and four-fifths of the farmers)
    to be about two shillings per annum, rags included; and I believe no
    gentleman would repine to give ten shillings for the carcass of a good
    fat child, which, as I have said, will make four dishes of excellent
    nutritive meat, when he hath only some particular friend, or his
    own family to dine with him. Thus the squire will learn to be a good
    landlord, and grow popular among his tenants, the mother will have eight
    shillings neat profit, and be fit for work till she produces another
    child.
    
    Those who are more thrifty (as I must confess the times require) may
    flea the carcass; the skin of which, artificially dressed, will make
    admirable gloves for ladies, and summer boots for fine gentlemen.
    
    As to our City of Dublin, shambles may be appointed for this purpose, in
    the most convenient parts of it, and butchers we may be assured will not
    be wanting; although I rather recommend buying the children alive, and
    dressing them hot from the knife, as we do roasting pigs.
    
    A very worthy person, a true lover of his country, and whose virtues
    I highly esteem, was lately pleased, in discoursing on this matter, to
    offer a refinement upon my scheme. He said, that many gentlemen of this
    kingdom, having of late destroyed their deer, he conceived that the
    want of venison might be well supply'd by the bodies of young lads and
    maidens, not exceeding fourteen years of age, nor under twelve; so great
    a number of both sexes in every country being now ready to starve for
    want of work and service: And these to be disposed of by their parents
    if alive, or otherwise by their nearest relations. But with due
    deference to so excellent a friend, and so deserving a patriot, I
    cannot be altogether in his sentiments; for as to the males, my American
    acquaintance assured me from frequent experience, that their flesh was
    generally tough and lean, like that of our school-boys, by continual
    exercise, and their taste disagreeable, and to fatten them would not
    answer the charge. Then as to the females, it would, I think, with
    humble submission, be a loss to the publick, because they soon would
    become breeders themselves: And besides, it is not improbable that some
    scrupulous people might be apt to censure such a practice, (although
    indeed very unjustly) as a little bordering upon cruelty, which, I
    confess, hath always been with me the strongest objection against any
    project, how well soever intended.
    
    But in order to justify my friend, he confessed, that this expedient
    was put into his head by the famous Salmanaazor, a native of the island
    Formosa, who came from thence to London, above twenty years ago, and in
    conversation told my friend, that in his country, when any young person
    happened to be put to death, the executioner sold the carcass to persons
    of quality, as a prime dainty; and that, in his time, the body of a
    plump girl of fifteen, who was crucified for an attempt to poison the
    Emperor, was sold to his imperial majesty's prime minister of state, and
    other great mandarins of the court in joints from the gibbet, at four
    hundred crowns. Neither indeed can I deny, that if the same use were
    made of several plump young girls in this town, who without one single
    groat to their fortunes, cannot stir abroad without a chair, and appear
    at a play-house and assemblies in foreign fineries which they never will
    pay for; the kingdom would not be the worse.
    
    Some persons of a desponding spirit are in great concern about that vast
    number of poor people, who are aged, diseased, or maimed; and I have
    been desired to employ my thoughts what course may be taken, to ease
    the nation of so grievous an incumbrance. But I am not in the least pain
    upon that matter, because it is very well known, that they are every day
    dying, and rotting, by cold and famine, and filth, and vermin, as fast
    as can be reasonably expected. And as to the young labourers, they
    are now in almost as hopeful a condition. They cannot get work, and
    consequently pine away from want of nourishment, to a degree, that if
    at any time they are accidentally hired to common labour, they have not
    strength to perform it, and thus the country and themselves are happily
    delivered from the evils to come.
    
    I have too long digressed, and therefore shall return to my subject. I
    think the advantages by the proposal which I have made are obvious and
    many, as well as of the highest importance.
    
    For first, as I have already observed, it would greatly lessen the
    number of Papists, with whom we are yearly over-run, being the principal
    breeders of the nation, as well as our most dangerous enemies, and who
    stay at home on purpose with a design to deliver the kingdom to the
    Pretender, hoping to take their advantage by the absence of so many good
    Protestants, who have chosen rather to leave their country, than stay at
    home and pay tithes against their conscience to an episcopal curate.
    
    Secondly, The poorer tenants will have something valuable of their own,
    which by law may be made liable to a distress, and help to pay their
    landlord's rent, their corn and cattle being already seized, and money a
    thing unknown.
    
    Thirdly, Whereas the maintainance of an hundred thousand children,
    from two years old, and upwards, cannot be computed at less than
    ten shillings a piece per annum, the nation's stock will be thereby
    encreased fifty thousand pounds per annum, besides the profit of a
    new dish, introduced to the tables of all gentlemen of fortune in the
    kingdom, who have any refinement in taste. And the money will circulate
    among our selves, the goods being entirely of our own growth and
    manufacture.
    
    Fourthly, The constant breeders, besides the gain of eight shillings
    sterling per annum by the sale of their children, will be rid of the
    charge of maintaining them after the first year.
    
    Fifthly, This food would likewise bring great custom to taverns,
    where the vintners will certainly be so prudent as to procure the best
    receipts for dressing it to perfection; and consequently have their
    houses frequented by all the fine gentlemen, who justly value themselves
    upon their knowledge in good eating; and a skilful cook, who understands
    how to oblige his guests, will contrive to make it as expensive as they
    please.
    
    Sixthly, This would be a great inducement to marriage, which all wise
    nations have either encouraged by rewards, or enforced by laws and
    penalties. It would encrease the care and tenderness of mothers towards
    their children, when they were sure of a settlement for life to the
    poor babes, provided in some sort by the publick, to their annual profit
    instead of expence. We should soon see an honest emulation among the
    married women, which of them could bring the fattest child to the
    market. Men would become as fond of their wives, during the time of
    their pregnancy, as they are now of their mares in foal, their cows in
    calf, or sow when they are ready to farrow; nor offer to beat or kick
    them (as is too frequent a practice) for fear of a miscarriage.
    
    Many other advantages might be enumerated. For instance, the addition
    of some thousand carcasses in our exportation of barrel'd beef: the
    propagation of swine's flesh, and improvement in the art of making good
    bacon, so much wanted among us by the great destruction of pigs,
    too frequent at our tables; which are no way comparable in taste or
    magnificence to a well grown, fat yearly child, which roasted whole will
    make a considerable figure at a Lord Mayor's feast, or any other publick
    entertainment. But this, and many others, I omit, being studious of
    brevity.
    
    Supposing that one thousand families in this city, would be constant
    customers for infants flesh, besides others who might have it at merry
    meetings, particularly at weddings and christenings, I compute that
    Dublin would take off annually about twenty thousand carcasses; and the
    rest of the kingdom (where probably they will be sold somewhat cheaper)
    the remaining eighty thousand.
    
    I can think of no one objection, that will possibly be raised against
    this proposal, unless it should be urged, that the number of people will
    be thereby much lessened in the kingdom. This I freely own, and 'twas
    indeed one principal design in offering it to the world. I desire the
    reader will observe, that I calculate my remedy for this one individual
    Kingdom of Ireland, and for no other that ever was, is, or, I think,
    ever can be upon Earth. Therefore let no man talk to me of other
    expedients: Of taxing our absentees at five shillings a pound: Of using
    neither cloaths, nor houshold furniture, except what is of our
    own growth and manufacture: Of utterly rejecting the materials and
    instruments that promote foreign luxury: Of curing the expensiveness of
    pride, vanity, idleness, and gaming in our women: Of introducing a vein
    of parsimony, prudence and temperance: Of learning to love our
    country, wherein we differ even from Laplanders, and the inhabitants
    of Topinamboo: Of quitting our animosities and factions, nor acting any
    longer like the Jews, who were murdering one another at the very moment
    their city was taken: Of being a little cautious not to sell our country
    and consciences for nothing: Of teaching landlords to have at least one
    degree of mercy towards their tenants. Lastly, of putting a spirit of
    honesty, industry, and skill into our shop-keepers, who, if a resolution
    could now be taken to buy only our native goods, would immediately unite
    to cheat and exact upon us in the price, the measure, and the goodness,
    nor could ever yet be brought to make one fair proposal of just dealing,
    though often and earnestly invited to it.
    
    Therefore I repeat, let no man talk to me of these and the like
    expedients, 'till he hath at least some glympse of hope, that there will
    ever be some hearty and sincere attempt to put them into practice.
    
    But, as to my self, having been wearied out for many years with offering
    vain, idle, visionary thoughts, and at length utterly despairing of
    success, I fortunately fell upon this proposal, which, as it is wholly
    new, so it hath something solid and real, of no expence and little
    trouble, full in our own power, and whereby we can incur no danger
    in disobliging England. For this kind of commodity will not bear
    exportation, and flesh being of too tender a consistence, to admit a
    long continuance in salt, although perhaps I could name a country, which
    would be glad to eat up our whole nation without it.
    
    After all, I am not so violently bent upon my own opinion, as to reject
    any offer, proposed by wise men, which shall be found equally innocent,
    cheap, easy, and effectual. But before something of that kind shall be
    advanced in contradiction to my scheme, and offering a better, I desire
    the author or authors will be pleased maturely to consider two points.
    First, As things now stand, how they will be able to find food and
    raiment for a hundred thousand useless mouths and backs. And secondly,
    There being a round million of creatures in humane figure throughout
    this kingdom, whose whole subsistence put into a common stock, would
    leave them in debt two million of pounds sterling, adding those who are
    beggars by profession, to the bulk of farmers, cottagers and labourers,
    with their wives and children, who are beggars in effect; I desire
    those politicians who dislike my overture, and may perhaps be so bold
    to attempt an answer, that they will first ask the parents of these
    mortals, whether they would not at this day think it a great happiness
    to have been sold for food at a year old, in the manner I prescribe, and
    thereby have avoided such a perpetual scene of misfortunes, as they have
    since gone through, by the oppression of landlords, the impossibility of
    paying rent without money or trade, the want of common sustenance, with
    neither house nor cloaths to cover them from the inclemencies of the
    weather, and the most inevitable prospect of intailing the like, or
    greater miseries, upon their breed for ever.
    
    I profess, in the sincerity of my heart, that I have not the least
    personal interest in endeavouring to promote this necessary work, having
    no other motive than the publick good of my country, by advancing
    our trade, providing for infants, relieving the poor, and giving some
    pleasure to the rich. I have no children, by which I can propose to
    get a single penny; the youngest being nine years old, and my wife past
    child-bearing.
    
    
    
    
    
    End of the Project Gutenberg EBook of A Modest Proposal, by Jonathan Swift
    
    *** END OF THIS PROJECT GUTENBERG EBOOK A MODEST PROPOSAL ***
    
    ***** This file should be named 1080.txt or 1080.zip *****
    This and all associated files of various formats will be found in:
            http://www.gutenberg.org/1/0/8/1080/
    
    Produced by An Anonymous Volunteer
    
    Updated editions will replace the previous one--the old editions
    will be renamed.
    
    Creating the works from public domain print editions means that no
    one owns a United States copyright in these works, so the Foundation
    (and you!) can copy and distribute it in the United States without
    permission and without paying copyright royalties.  Special rules,
    set forth in the General Terms of Use part of this license, apply to
    copying and distributing Project Gutenberg-tm electronic works to
    protect the PROJECT GUTENBERG-tm concept and trademark.  Project
    Gutenberg is a registered trademark, and may not be used if you
    charge for the eBooks, unless you receive specific permission.  If you
    do not charge anything for copies of this eBook, complying with the
    rules is very easy.  You may use this eBook for nearly any purpose
    such as creation of derivative works, reports, performances and
    research.  They may be modified and printed and given away--you may do
    practically ANYTHING with public domain eBooks.  Redistribution is
    subject to the trademark license, especially commercial
    redistribution.
    
    
    
    *** START: FULL LICENSE ***
    
    THE FULL PROJECT GUTENBERG LICENSE
    PLEASE READ THIS BEFORE YOU DISTRIBUTE OR USE THIS WORK
    
    To protect the Project Gutenberg-tm mission of promoting the free
    distribution of electronic works, by using or distributing this work
    (or any other work associated in any way with the phrase "Project
    Gutenberg"), you agree to comply with all the terms of the Full Project
    Gutenberg-tm License (available with this file or online at
    http://gutenberg.org/license).
    
    
    Section 1.  General Terms of Use and Redistributing Project Gutenberg-tm
    electronic works
    
    1.A.  By reading or using any part of this Project Gutenberg-tm
    electronic work, you indicate that you have read, understand, agree to
    and accept all the terms of this license and intellectual property
    (trademark/copyright) agreement.  If you do not agree to abide by all
    the terms of this agreement, you must cease using and return or destroy
    all copies of Project Gutenberg-tm electronic works in your possession.
    If you paid a fee for obtaining a copy of or access to a Project
    Gutenberg-tm electronic work and you do not agree to be bound by the
    terms of this agreement, you may obtain a refund from the person or
    entity to whom you paid the fee as set forth in paragraph 1.E.8.
    
    1.B.  "Project Gutenberg" is a registered trademark.  It may only be
    used on or associated in any way with an electronic work by people who
    agree to be bound by the terms of this agreement.  There are a few
    things that you can do with most Project Gutenberg-tm electronic works
    even without complying with the full terms of this agreement.  See
    paragraph 1.C below.  There are a lot of things you can do with Project
    Gutenberg-tm electronic works if you follow the terms of this agreement
    and help preserve free future access to Project Gutenberg-tm electronic
    works.  See paragraph 1.E below.
    
    1.C.  The Project Gutenberg Literary Archive Foundation ("the Foundation"
    or PGLAF), owns a compilation copyright in the collection of Project
    Gutenberg-tm electronic works.  Nearly all the individual works in the
    collection are in the public domain in the United States.  If an
    individual work is in the public domain in the United States and you are
    located in the United States, we do not claim a right to prevent you from
    copying, distributing, performing, displaying or creating derivative
    works based on the work as long as all references to Project Gutenberg
    are removed.  Of course, we hope that you will support the Project
    Gutenberg-tm mission of promoting free access to electronic works by
    freely sharing Project Gutenberg-tm works in compliance with the terms of
    this agreement for keeping the Project Gutenberg-tm name associated with
    the work.  You can easily comply with the terms of this agreement by
    keeping this work in the same format with its attached full Project
    Gutenberg-tm License when you share it without charge with others.
    
    1.D.  The copyright laws of the place where you are located also govern
    what you can do with this work.  Copyright laws in most countries are in
    a constant state of change.  If you are outside the United States, check
    the laws of your country in addition to the terms of this agreement
    before downloading, copying, displaying, performing, distributing or
    creating derivative works based on this work or any other Project
    Gutenberg-tm work.  The Foundation makes no representations concerning
    the copyright status of any work in any country outside the United
    States.
    
    1.E.  Unless you have removed all references to Project Gutenberg:
    
    1.E.1.  The following sentence, with active links to, or other immediate
    access to, the full Project Gutenberg-tm License must appear prominently
    whenever any copy of a Project Gutenberg-tm work (any work on which the
    phrase "Project Gutenberg" appears, or with which the phrase "Project
    Gutenberg" is associated) is accessed, displayed, performed, viewed,
    copied or distributed:
    
    This eBook is for the use of anyone anywhere at no cost and with
    almost no restrictions whatsoever.  You may copy it, give it away or
    re-use it under the terms of the Project Gutenberg License included
    with this eBook or online at www.gutenberg.org
    
    1.E.2.  If an individual Project Gutenberg-tm electronic work is derived
    from the public domain (does not contain a notice indicating that it is
    posted with permission of the copyright holder), the work can be copied
    and distributed to anyone in the United States without paying any fees
    or charges.  If you are redistributing or providing access to a work
    with the phrase "Project Gutenberg" associated with or appearing on the
    work, you must comply either with the requirements of paragraphs 1.E.1
    through 1.E.7 or obtain permission for the use of the work and the
    Project Gutenberg-tm trademark as set forth in paragraphs 1.E.8 or
    1.E.9.
    
    1.E.3.  If an individual Project Gutenberg-tm electronic work is posted
    with the permission of the copyright holder, your use and distribution
    must comply with both paragraphs 1.E.1 through 1.E.7 and any additional
    terms imposed by the copyright holder.  Additional terms will be linked
    to the Project Gutenberg-tm License for all works posted with the
    permission of the copyright holder found at the beginning of this work.
    
    1.E.4.  Do not unlink or detach or remove the full Project Gutenberg-tm
    License terms from this work, or any files containing a part of this
    work or any other work associated with Project Gutenberg-tm.
    
    1.E.5.  Do not copy, display, perform, distribute or redistribute this
    electronic work, or any part of this electronic work, without
    prominently displaying the sentence set forth in paragraph 1.E.1 with
    active links or immediate access to the full terms of the Project
    Gutenberg-tm License.
    
    1.E.6.  You may convert to and distribute this work in any binary,
    compressed, marked up, nonproprietary or proprietary form, including any
    word processing or hypertext form.  However, if you provide access to or
    distribute copies of a Project Gutenberg-tm work in a format other than
    "Plain Vanilla ASCII" or other format used in the official version
    posted on the official Project Gutenberg-tm web site (www.gutenberg.org),
    you must, at no additional cost, fee or expense to the user, provide a
    copy, a means of exporting a copy, or a means of obtaining a copy upon
    request, of the work in its original "Plain Vanilla ASCII" or other
    form.  Any alternate format must include the full Project Gutenberg-tm
    License as specified in paragraph 1.E.1.
    
    1.E.7.  Do not charge a fee for access to, viewing, displaying,
    performing, copying or distributing any Project Gutenberg-tm works
    unless you comply with paragraph 1.E.8 or 1.E.9.
    
    1.E.8.  You may charge a reasonable fee for copies of or providing
    access to or distributing Project Gutenberg-tm electronic works provided
    that
    
    - You pay a royalty fee of 20% of the gross profits you derive from
         the use of Project Gutenberg-tm works calculated using the method
         you already use to calculate your applicable taxes.  The fee is
         owed to the owner of the Project Gutenberg-tm trademark, but he
         has agreed to donate royalties under this paragraph to the
         Project Gutenberg Literary Archive Foundation.  Royalty payments
         must be paid within 60 days following each date on which you
         prepare (or are legally required to prepare) your periodic tax
         returns.  Royalty payments should be clearly marked as such and
         sent to the Project Gutenberg Literary Archive Foundation at the
         address specified in Section 4, "Information about donations to
         the Project Gutenberg Literary Archive Foundation."
    
    - You provide a full refund of any money paid by a user who notifies
         you in writing (or by e-mail) within 30 days of receipt that s/he
         does not agree to the terms of the full Project Gutenberg-tm
         License.  You must require such a user to return or
         destroy all copies of the works possessed in a physical medium
         and discontinue all use of and all access to other copies of
         Project Gutenberg-tm works.
    
    - You provide, in accordance with paragraph 1.F.3, a full refund of any
         money paid for a work or a replacement copy, if a defect in the
         electronic work is discovered and reported to you within 90 days
         of receipt of the work.
    
    - You comply with all other terms of this agreement for free
         distribution of Project Gutenberg-tm works.
    
    1.E.9.  If you wish to charge a fee or distribute a Project Gutenberg-tm
    electronic work or group of works on different terms than are set
    forth in this agreement, you must obtain permission in writing from
    both the Project Gutenberg Literary Archive Foundation and Michael
    Hart, the owner of the Project Gutenberg-tm trademark.  Contact the
    Foundation as set forth in Section 3 below.
    
    1.F.
    
    1.F.1.  Project Gutenberg volunteers and employees expend considerable
    effort to identify, do copyright research on, transcribe and proofread
    public domain works in creating the Project Gutenberg-tm
    collection.  Despite these efforts, Project Gutenberg-tm electronic
    works, and the medium on which they may be stored, may contain
    "Defects," such as, but not limited to, incomplete, inaccurate or
    corrupt data, transcription errors, a copyright or other intellectual
    property infringement, a defective or damaged disk or other medium, a
    computer virus, or computer codes that damage or cannot be read by
    your equipment.
    
    1.F.2.  LIMITED WARRANTY, DISCLAIMER OF DAMAGES - Except for the "Right
    of Replacement or Refund" described in paragraph 1.F.3, the Project
    Gutenberg Literary Archive Foundation, the owner of the Project
    Gutenberg-tm trademark, and any other party distributing a Project
    Gutenberg-tm electronic work under this agreement, disclaim all
    liability to you for damages, costs and expenses, including legal
    fees.  YOU AGREE THAT YOU HAVE NO REMEDIES FOR NEGLIGENCE, STRICT
    LIABILITY, BREACH OF WARRANTY OR BREACH OF CONTRACT EXCEPT THOSE
    PROVIDED IN PARAGRAPH F3.  YOU AGREE THAT THE FOUNDATION, THE
    TRADEMARK OWNER, AND ANY DISTRIBUTOR UNDER THIS AGREEMENT WILL NOT BE
    LIABLE TO YOU FOR ACTUAL, DIRECT, INDIRECT, CONSEQUENTIAL, PUNITIVE OR
    INCIDENTAL DAMAGES EVEN IF YOU GIVE NOTICE OF THE POSSIBILITY OF SUCH
    DAMAGE.
    
    1.F.3.  LIMITED RIGHT OF REPLACEMENT OR REFUND - If you discover a
    defect in this electronic work within 90 days of receiving it, you can
    receive a refund of the money (if any) you paid for it by sending a
    written explanation to the person you received the work from.  If you
    received the work on a physical medium, you must return the medium with
    your written explanation.  The person or entity that provided you with
    the defective work may elect to provide a replacement copy in lieu of a
    refund.  If you received the work electronically, the person or entity
    providing it to you may choose to give you a second opportunity to
    receive the work electronically in lieu of a refund.  If the second copy
    is also defective, you may demand a refund in writing without further
    opportunities to fix the problem.
    
    1.F.4.  Except for the limited right of replacement or refund set forth
    in paragraph 1.F.3, this work is provided to you 'AS-IS' WITH NO OTHER
    WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    WARRANTIES OF MERCHANTIBILITY OR FITNESS FOR ANY PURPOSE.
    
    1.F.5.  Some states do not allow disclaimers of certain implied
    warranties or the exclusion or limitation of certain types of damages.
    If any disclaimer or limitation set forth in this agreement violates the
    law of the state applicable to this agreement, the agreement shall be
    interpreted to make the maximum disclaimer or limitation permitted by
    the applicable state law.  The invalidity or unenforceability of any
    provision of this agreement shall not void the remaining provisions.
    
    1.F.6.  INDEMNITY - You agree to indemnify and hold the Foundation, the
    trademark owner, any agent or employee of the Foundation, anyone
    providing copies of Project Gutenberg-tm electronic works in accordance
    with this agreement, and any volunteers associated with the production,
    promotion and distribution of Project Gutenberg-tm electronic works,
    harmless from all liability, costs and expenses, including legal fees,
    that arise directly or indirectly from any of the following which you do
    or cause to occur: (a) distribution of this or any Project Gutenberg-tm
    work, (b) alteration, modification, or additions or deletions to any
    Project Gutenberg-tm work, and (c) any Defect you cause.
    
    
    Section  2.  Information about the Mission of Project Gutenberg-tm
    
    Project Gutenberg-tm is synonymous with the free distribution of
    electronic works in formats readable by the widest variety of computers
    including obsolete, old, middle-aged and new computers.  It exists
    because of the efforts of hundreds of volunteers and donations from
    people in all walks of life.
    
    Volunteers and financial support to provide volunteers with the
    assistance they need, is critical to reaching Project Gutenberg-tm's
    goals and ensuring that the Project Gutenberg-tm collection will
    remain freely available for generations to come.  In 2001, the Project
    Gutenberg Literary Archive Foundation was created to provide a secure
    and permanent future for Project Gutenberg-tm and future generations.
    To learn more about the Project Gutenberg Literary Archive Foundation
    and how your efforts and donations can help, see Sections 3 and 4
    and the Foundation web page at http://www.pglaf.org.
    
    
    Section 3.  Information about the Project Gutenberg Literary Archive
    Foundation
    
    The Project Gutenberg Literary Archive Foundation is a non profit
    501(c)(3) educational corporation organized under the laws of the
    state of Mississippi and granted tax exempt status by the Internal
    Revenue Service.  The Foundation's EIN or federal tax identification
    number is 64-6221541.  Its 501(c)(3) letter is posted at
    http://pglaf.org/fundraising.  Contributions to the Project Gutenberg
    Literary Archive Foundation are tax deductible to the full extent
    permitted by U.S. federal laws and your state's laws.
    
    The Foundation's principal office is located at 4557 Melan Dr. S.
    Fairbanks, AK, 99712., but its volunteers and employees are scattered
    throughout numerous locations.  Its business office is located at
    809 North 1500 West, Salt Lake City, UT 84116, (801) 596-1887, email
    business@pglaf.org.  Email contact links and up to date contact
    information can be found at the Foundation's web site and official
    page at http://pglaf.org
    
    For additional contact information:
         Dr. Gregory B. Newby
         Chief Executive and Director
         gbnewby@pglaf.org
    
    
    Section 4.  Information about Donations to the Project Gutenberg
    Literary Archive Foundation
    
    Project Gutenberg-tm depends upon and cannot survive without wide
    spread public support and donations to carry out its mission of
    increasing the number of public domain and licensed works that can be
    freely distributed in machine readable form accessible by the widest
    array of equipment including outdated equipment.  Many small donations
    ($1 to $5,000) are particularly important to maintaining tax exempt
    status with the IRS.
    
    The Foundation is committed to complying with the laws regulating
    charities and charitable donations in all 50 states of the United
    States.  Compliance requirements are not uniform and it takes a
    considerable effort, much paperwork and many fees to meet and keep up
    with these requirements.  We do not solicit donations in locations
    where we have not received written confirmation of compliance.  To
    SEND DONATIONS or determine the status of compliance for any
    particular state visit http://pglaf.org
    
    While we cannot and do not solicit contributions from states where we
    have not met the solicitation requirements, we know of no prohibition
    against accepting unsolicited donations from donors in such states who
    approach us with offers to donate.
    
    International donations are gratefully accepted, but we cannot make
    any statements concerning tax treatment of donations received from
    outside the United States.  U.S. laws alone swamp our small staff.
    
    Please check the Project Gutenberg Web pages for current donation
    methods and addresses.  Donations are accepted in a number of other
    ways including checks, online payments and credit card donations.
    To donate, please visit: http://pglaf.org/donate
    
    
    Section 5.  General Information About Project Gutenberg-tm electronic
    works.
    
    Professor Michael S. Hart is the originator of the Project Gutenberg-tm
    concept of a library of electronic works that could be freely shared
    with anyone.  For thirty years, he produced and distributed Project
    Gutenberg-tm eBooks with only a loose network of volunteer support.
    
    
    Project Gutenberg-tm eBooks are often created from several printed
    editions, all of which are confirmed as Public Domain in the U.S.
    unless a copyright notice is included.  Thus, we do not necessarily
    keep eBooks in compliance with any particular paper edition.
    
    
    Most people start at our Web site which has the main PG search facility:
    
         http://www.gutenberg.org
    
    This Web site includes information about Project Gutenberg-tm,
    including how to make donations to the Project Gutenberg Literary
    Archive Foundation, how to help produce our new eBooks, and how to
    subscribe to our email newsletter to hear about new eBooks.
    


The books were were working with yesterday had already had some processing done on them so that we could use NLTK to find features of the language. Remember that Python regards a text file as a single long string of characters. The first thing to do is to start breaking the text up into sentences and words.


```python
from nltk import word_tokenize
text = open('/home/researcher/modest_proposal.txt', "r", encoding='UTF-8').read() 
tokens = word_tokenize(text)
print(tokens[:100])
```

    ['\ufeffThe', 'Project', 'Gutenberg', 'EBook', 'of', 'A', 'Modest', 'Proposal', ',', 'by', 'Jonathan', 'Swift', 'This', 'eBook', 'is', 'for', 'the', 'use', 'of', 'anyone', 'anywhere', 'at', 'no', 'cost', 'and', 'with', 'almost', 'no', 'restrictions', 'whatsoever', '.', 'You', 'may', 'copy', 'it', ',', 'give', 'it', 'away', 'or', 're-use', 'it', 'under', 'the', 'terms', 'of', 'the', 'Project', 'Gutenberg', 'License', 'included', 'with', 'this', 'eBook', 'or', 'online', 'at', 'www.gutenberg.org', 'Title', ':', 'A', 'Modest', 'Proposal', 'For', 'preventing', 'the', 'children', 'of', 'poor', 'people', 'in', 'Ireland', ',', 'from', 'being', 'a', 'burden', 'on', 'their', 'parents', 'or', 'country', ',', 'and', 'for', 'making', 'them', 'beneficial', 'to', 'the', 'publick', '-', '1729', 'Author', ':', 'Jonathan', 'Swift', 'Posting', 'Date', ':']


**Challenge!**

1. Find a .txt file from the Gutenberg Project or elsewhere and upload it to the Jupyter Notebook. 
2. Use the word_tokenize to break up the text data. 
3. Print the first 100 tokens.

Breaking a speech into tokens lets us do the sort of word counting that we were doing yesterday on the speeches. We can do some more interesting linguistic analysis if we use Part of Speech tagging. NLTK has a number of different Part of Speech tags that we could use, but the simplest one is called 'Universal', and we'll use that here.


```python
sentence = "They refuse to permit us the refuse permit"
words = word_tokenize(sentence)
tagged = nltk.pos_tag(words, tagset='universal')
print(tagged)
```

    [('They', u'PRON'), ('refuse', u'VERB'), ('to', u'PRT'), ('permit', u'VERB'), ('us', u'PRON'), ('the', u'DET'), ('refuse', u'NOUN'), ('permit', u'NOUN')]


Part of Speech tagging creates bigrams, that is, it associates the word with its tag in a pair of items that we can see above in brackets.  


```python
tag_fd = nltk.FreqDist(tag for (word, tag) in tagged)
tag_fd.most_common()
```




    [(u'PRON', 2), (u'VERB', 2), (u'NOUN', 2), (u'DET', 1), (u'PRT', 1)]



**Challenge!**

Use Part of Speech tagging to tag the text that we have just tokenised the do the following:
* Find the most common parts of speech
* Find the most common verbs and create a frequency Distribution graph of your result
* Find the 10 most common nouns in the text

*Hint: to find the most common verbs and nouns, you will need to create a list that contains only the verbs or only the nouns from the speech. Use a for loop to create your list. Then create a frequency distribution*


```python
tagged_text = nltk.pos_tag(tokens, tagset = 'universal')
text_fd = nltk.FreqDist(tag for (word, tag) in tagged_text)
text_fd.most_common()
```




    [('NOUN', 1871),
     ('VERB', 1082),
     ('ADP', 905),
     ('.', 846),
     ('DET', 751),
     ('ADJ', 503),
     ('PRON', 399),
     ('CONJ', 314),
     ('ADV', 296),
     ('PRT', 208),
     ('NUM', 129)]




```python
verblist = []
for (word, tag) in tagged_text:
    if tag == 'VERB':
        verblist.append(word)
# Check the length of the list of verbs. 
#If it matches the number of verbs above, you can be fairly sure your loop has worked as expected
print(len(verblist))
verb_fd = nltk.FreqDist(verblist)
print(verb_fd.most_common()[:10])
```

    1082
    [('be', 67), ('is', 47), ('are', 41), ('will', 41), ('have', 33), ('can', 29), ('may', 26), ('would', 18), ('being', 17), ('do', 16)]



```python
nounlist = []
for (word, tag) in tagged_text:
    if tag == 'NOUN':
        nounlist.append(word)
print(nounlist[:10])
print(len(nounlist))
noun_fd = nltk.FreqDist(nounlist)
print(noun_fd.most_common()[:10])
```

    ['Project', 'Gutenberg', 'EBook', 'A', 'Modest', 'Proposal', 'Jonathan', 'Swift', 'eBook', 'use']
    1871
    [('Project', 80), ('Gutenberg-tm', 55), ('work', 49), ('works', 30), ('Gutenberg', 27), ('Foundation', 24), ('children', 20), ('terms', 19), ('agreement', 17), ('country', 16)]


**Extension**
There are a few things to note about this result - Project and Gutenberg have been returned as two different, very frequent nouns. Because we're humans, not computers, we know it's likely that they are often occuring together. We could test for bigrams (words that typically occur side by side) to see if this is the case. 

In order to perform this test, we must first convert our list of tokens into and NLTK text. We can then use specific NLTK functions on the text.


```python
print(type(tokens))
nltk_text = nltk.Text(tokens)
print(type(nltk_text))
nltk_text.collocations()
```

    <class 'list'>
    <class 'nltk.text.Text'>
    Project Gutenberg-tm; Project Gutenberg; Literary Archive; Archive
    Foundation; Gutenberg Literary; United States; Gutenberg-tm
    electronic; electronic works; set forth; public domain; electronic
    work; Gutenberg-tm License; Jonathan Swift; per annum; copyright
    holder; MODEST PROPOSAL; Modest Proposal; PROJECT GUTENBERG; twenty
    thousand; year old


### Some linguistics...

*Functional linguistics* is a research area concerned with how *realised language* (lexis and grammar) work to achieve meaningful social functions.

One functional linguistic theory is *Systemic Functional Linguistics*, developed by Michael Halliday (Prof. Emeritus at University of Sydney).

Central to the theory is a division between **experiential meanings** and **interpersonal meanings**.

* Experiential meanings communicate what happened to whom, under what circumstances.
* Interpersonal meanings negotiate identities and role relationships between speakers 

Halliday argues that these two kinds of meaning are realised **simultaneously** through different parts of English grammar.

* Experiential meanings are made through **transitivity choices**.
* Interpersonal meanings are made through **mood choices**


Transitivity choices include fitting together configurations of:

* Participants (*a man, green bikes*)
* Processes (*sleep, has always been, is considering*)
* Circumstances (*on the weekend*, *in Australia*)

Mood features of a language include:

* Mood types (*declarative, interrogative, imperative*)
* Modality (*would, can, might*)
* Lexical density--wordshe number of words per clause, the number of content to non-content words, etc.

Lexical density is usually a good indicator of the general tone of texts. The language of academia, for example, often has a huge number of nouns to verbs. We can approximate an academic tone simply by making nominally dense clauses: 

      The consideration of interest is the potential for a participant of a certain demographic to be in Group A or Group B*.

Notice how not only are there many nouns (*consideration*, *interest*, *potential*, etc.), but that the verbs are very simple (*is*, *to be*).

In comparison, informal speech is characterised by smaller clauses, and thus more verbs.

      A: Did you feel like dropping by?
      B: I thought I did, but now I don't think I want to

Here, we have only a few, simple nouns (*you*, *I*), with more expressive verbs (*feel*, *dropping by*, *think*, *want*)

> **Note**: SFL argues that through *grammatical metaphor*, one linguistic feature can stand in for another. *Would you please shut the door?* is an interrogative, but it functions as a command. *invitation* is a nominalisation of a process, *invite*. We don't have time to deal with these kinds of realisations, unfortunately.

In the context of Fraser's speech, there are nearly twice as many nouns as verbs, and the verbs are generally quite simple ones (parts of To Be and To Have make up about a quarter). This suggests that Fraser's speech, even when giving a radio talk to his electorate, is more towards the formal end of the spectrum. 

## Recap
So far today we have:
* Imported text into NLTK
* Tokenised raw text into words
* Tagged words as parts of speech
* Converted a list into NLTK Text for further analysis

## Stopwords
Yesterday, when we did our frequency counts of the books in the NLTK Library, we noticed that a lot of speace was taken up by little words like 'and' and 'of' and 'the' which don't add a lot to our understanding of text. These are called 'stop words'. It will help our analysis if we exclude them.


```python
fdist1 = nltk.FreqDist(tokens)
fdist1.most_common()[:20]
```




    [(',', 510),
     ('the', 327),
     ('of', 236),
     ('.', 183),
     ('to', 182),
     ('and', 177),
     ('a', 141),
     ('in', 125),
     ('or', 106),
     ('Project', 83),
     ('be', 67),
     ('for', 63),
     ('this', 60),
     ('with', 58),
     ('by', 55),
     ('Gutenberg-tm', 55),
     ('I', 55),
     ('you', 52),
     ('that', 51),
     ('work', 50)]




```python
print(len(nltk_text))
print(len(set(nltk_text)))
```

    7304
    1795



```python
#First let's get rid of the puncutation
text = [word for word in nltk_text if word.isalpha()]
print(len(text))#Then get rid of capitals
vocab = [word.lower() for word in text]
print(len(set(vocab)))
```

    6273
    1551



```python
from nltk.corpus import stopwords
#Create a variable that contains all the stopwords in the NLTK corpus
ignored_words = nltk.corpus.stopwords.words('english')
unstopped = [word for word in vocab if word not in stopwords.words('english')]
fdist2 = nltk.FreqDist(unstopped)
fdist2.most_common()[:20]
```




    [('project', 88),
     ('work', 51),
     ('works', 32),
     ('gutenberg', 30),
     ('electronic', 27),
     ('may', 26),
     ('foundation', 25),
     ('terms', 21),
     ('children', 20),
     ('agreement', 18),
     ('would', 18),
     ('one', 17),
     ('country', 16),
     ('thousand', 15),
     ('kingdom', 15),
     ('donations', 15),
     ('upon', 15),
     ('license', 15),
     ('states', 14),
     ('number', 14)]



The list we have now is probably more intersting if we wanted to get a sense of the key issues in the text. Note, we're working with a very small sample here. This sort of analysis is much more useful over really big corpora.

*Note: We could have condensed the first two steps into a single line of code that looked like this:*

        unstopped = [word for word in speech if word.lower() not in stopwords.words('english') and word.isalpha()]

## Collocation
We've just used collocation to test a hypothesis about the most common nouns in the speech we were investigating. Collocation can be quite a powerful tool for finding features of language.

First, let's look for bigrams in the whole list of tokens:


```python
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(tokens)
sorted(finder.nbest(bigram_measures.raw_freq, 10))
```




    [(',', 'and'),
     (',', 'or'),
     (',', 'that'),
     (',', 'the'),
     ('Project', 'Gutenberg'),
     ('Project', 'Gutenberg-tm'),
     ('in', 'the'),
     ('of', 'the'),
     ('the', 'Project'),
     ('to', 'the')]



That doesn't tell us much. Let's try again with 'unstopped' our list of tokens with the punctuation and stopwords removed


```python
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(unstopped)
sorted(finder.nbest(bigram_measures.raw_freq, 10))
```




    [('archive', 'foundation'),
     ('electronic', 'work'),
     ('electronic', 'works'),
     ('gutenberg', 'literary'),
     ('literary', 'archive'),
     ('project', 'electronic'),
     ('project', 'gutenberg'),
     ('project', 'license'),
     ('terms', 'agreement'),
     ('united', 'states')]



As well as identifying collocations (words that appear near each other), we can also look for n-grams or clusters, which appear immediately adjacent to each other. Repeated N-grams are a good way to get a sense of what a text is about. First, let's see how n-grams are created:


```python
print(sent2)
```

    ['The', 'family', 'of', 'Dashwood', 'had', 'long', 'been', 'settled', 'in', 'Sussex', '.']



```python
from nltk.util import ngrams
trigrams = ngrams(sent2, 3)
for gram in trigrams:
    print(gram)
```

    ('The', 'family', 'of')
    ('family', 'of', 'Dashwood')
    ('of', 'Dashwood', 'had')
    ('Dashwood', 'had', 'long')
    ('had', 'long', 'been')
    ('long', 'been', 'settled')
    ('been', 'settled', 'in')
    ('settled', 'in', 'Sussex')
    ('in', 'Sussex', '.')


There are a lot of trigrams in the sentence, and they don't tell us much. It's when n-grams are repeated that they start to get interesting, but before we write code the code for that we need to have some knowledge of dictionaries...

### Building a dictionaries

We've already worked with strings and lists. Another kind of data structure in Python is a dictionary.
Here is how a simple dictionary works:


```python
# create a dictionary
commonwords = {'the': 4023, 'of': 3809, 'a': 3098}
# search the dictionary for 'of'
commonwords['of']
```




    3809




```python
type(commonwords)
```




    dict



The point of dictionaries is to store a key (the word) and a value (the count). When you ask for the key, you get its value.

Notice that you use curly braces for dictionaries, but square brackets for lists.

### Finding duplicate ngrams


```python
import operator
from collections import Counter
threshold = 2
ng = 3
testtext = tokens

#Create out ngram, convert to a list, 
#run a counter to count the number of entries for each unique list element
raw_grams = ngrams(testtext, ng)
listgrams = list(raw_grams)
counts = Counter(listgrams)
print(len(listgrams), len(counts))
#Create a regular dictionary, this is mostly done so we can ignore Counter values less than threshold
D = {}
for k,v in counts.items():
    if v > threshold:
        D[k] = v
#Here is a way to sort a dictionary, based on the value (key=operator.itemgetter(1))
sorted_x = sorted(D.items(), key=operator.itemgetter(1), reverse=True)
```

    7302 6540



```python
sorted_x
```




    [(('Project', 'Gutenberg-tm', 'electronic'), 18),
     (('the', 'Project', 'Gutenberg'), 15),
     (('Project', 'Gutenberg', 'Literary'), 13),
     (('Gutenberg', 'Literary', 'Archive'), 13),
     (('Literary', 'Archive', 'Foundation'), 13),
     (('Gutenberg-tm', 'electronic', 'works'), 12),
     (('the', 'Project', 'Gutenberg-tm'), 12),
     (('the', 'terms', 'of'), 12),
     (('terms', 'of', 'this'), 10),
     (('of', 'this', 'agreement'), 10),
     (('the', 'United', 'States'), 9),
     (('set', 'forth', 'in'), 8),
     (('.', 'If', 'you'), 8),
     (('of', 'Project', 'Gutenberg-tm'), 8),
     (('Project', 'Gutenberg-tm', 'License'), 8),
     (('of', 'the', 'Project'), 8),
     (('to', 'the', 'Project'), 7),
     (('Gutenberg-tm', 'electronic', 'work'), 6),
     (('this', 'agreement', ','), 6),
     (('terms', 'of', 'the'), 5),
     ((',', 'you', 'must'), 5),
     (('.', 'You', 'may'), 5),
     (('Project', 'Gutenberg', "''"), 5),
     (('``', 'Project', 'Gutenberg'), 5),
     (('.', 'I', 'have'), 5),
     (('full', 'Project', 'Gutenberg-tm'), 5),
     (('in', 'the', 'United'), 5),
     (('a', 'Project', 'Gutenberg-tm'), 5),
     (('the', 'number', 'of'), 5),
     (('Project', 'Gutenberg-tm', 'works'), 5),
     (('Project', 'Gutenberg-tm', 'work'), 5),
     (('the', 'phrase', '``'), 4),
     (('Project', 'Gutenberg-tm', 'trademark'), 4),
     (('phrase', '``', 'Project'), 4),
     (('United', 'States', '.'), 4),
     (('the', 'copyright', 'holder'), 4),
     (('.', 'The', 'Foundation'), 4),
     (('the', 'full', 'Project'), 4),
     ((',', 'as', 'they'), 4),
     ((',', 'who', 'are'), 4),
     ((',', 'and', 'the'), 4),
     (('part', 'of', 'this'), 4),
     (('can', 'not', 'be'), 4),
     (('of', 'the', 'kingdom'), 4),
     (('or', 'any', 'other'), 4),
     (('the', 'use', 'of'), 4),
     (('at', 'http', ':'), 4),
     (('outside', 'the', 'United'), 3),
     ((',', 'as', 'I'), 3),
     (('If', 'an', 'individual'), 3),
     ((',', 'or', 'any'), 3),
     (('not', 'agree', 'to'), 3),
     (('as', 'set', 'forth'), 3),
     (('of', 'the', 'copyright'), 3),
     (('a', 'year', 'old'), 3),
     (('of', 'the', 'work'), 3),
     (('paragraph', '1.F.3', ','), 3),
     (('at', 'a', 'year'), 3),
     (('you', 'do', 'not'), 3),
     (('(', 'c', ')'), 3),
     (('.', 'Information', 'about'), 3),
     (('work', ',', 'or'), 3),
     (('the', 'owner', 'of'), 3),
     (('of', 'electronic', 'works'), 3),
     (('for', 'the', 'use'), 3),
     (('agreement', ',', 'you'), 3),
     ((',', 'and', 'for'), 3),
     (('received', 'the', 'work'), 3),
     (('free', 'distribution', 'of'), 3),
     (('by', 'all', 'the'), 3),
     (('year', 'old', ','), 3),
     (('.', 'If', 'an'), 3),
     ((',', 'that', 'a'), 3),
     (('you', 'received', 'the'), 3),
     (('this', 'kingdom', ','), 3),
     ((',', 'in', 'the'), 3),
     (('owner', 'of', 'the'), 3),
     (('this', 'electronic', 'work'), 3),
     (('A', 'MODEST', 'PROPOSAL'), 3),
     (('.', 'Do', 'not'), 3),
     (('electronic', 'work', 'is'), 3),
     (('I', 'have', 'already'), 3),
     (('permission', 'of', 'the'), 3),
     (('A', 'Modest', 'Proposal'), 3),
     (('children', 'of', 'poor'), 3),
     (('person', 'or', 'entity'), 3),
     (('of', 'poor', 'people'), 3),
     (('the', 'publick', ','), 3),
     (('electronic', 'works', '.'), 3),
     ((',', 'which', ','), 3),
     (('which', ',', 'as'), 3),
     (('forth', 'in', 'paragraph'), 3),
     (('you', 'can', 'do'), 3),
     (('any', 'Project', 'Gutenberg-tm'), 3),
     (('the', 'kingdom', ','), 3),
     (('can', 'do', 'with'), 3),
     (('to', 'the', 'publick'), 3),
     (('electronic', 'works', 'in'), 3),
     (('electronic', 'work', ','), 3),
     (('as', 'I', 'have'), 3),
     (('country', ',', 'and'), 3),
     (('The', 'Project', 'Gutenberg'), 3),
     (('years', 'old', ','), 3),
     ((';', 'and', 'I'), 3),
     ((',', 'as', 'to'), 3),
     ((',', 'performing', ','), 3),
     (('the', 'charge', 'of'), 3),
     (('all', 'the', 'terms'), 3),
     (('the', 'public', 'domain'), 3),
     (('I', 'have', 'been'), 3),
     (('electronic', 'works', ','), 3),
     ((',', 'and', 'therefore'), 3),
     (('per', 'annum', ','), 3),
     (('or', 'online', 'at'), 3),
     (('copies', 'of', 'Project'), 3),
     (('the', 'children', 'of'), 3),
     (('as', 'to', 'the'), 3),
     (('complying', 'with', 'the'), 3)]



This last bit of code is more advanced. Don't worry if you forget what every line means. If you are interested getting more comfortable with Python, come to our [Python]('https://github.com/resbaz/2015-12-14-Python-for-Researchers') course.

# Web scraping using Beautiful Soup

The most important skill for using NLTK in your life as a researchers is going to be working with your own texts. First, let's look at reading in text files directly from the web.

Of course, a lot of the text you're going to want to work with won't be in handy text files already. That's where a Python library called Beautiful Soup comes in.

*Note*: the ! is a way of accessing command line functions from the notebook. We could also do this in the terminal (without the !). 


```python
!sudo pip3 install BeautifulSoup4
from urllib.request import urlopen
```

    Requirement already satisfied (use --upgrade to upgrade): BeautifulSoup4 in /usr/local/lib/python3.4/dist-packages
    Cleaning up...



```python
from bs4 import BeautifulSoup
```


```python
url = "http://en.wikipedia.org/wiki/Smog"
```


```python
raw = urlopen(url).read()
print(type(raw))
print(raw[100:200])
```

    <class 'bytes'>
    b'>Smog - Wikipedia, the free encyclopedia</title>\n<script>document.documentElement.className = docume'


Beautiful Soup breaks the single long string into its constituent parts, creating an object 'Beautiful Soup'


```python
soup = BeautifulSoup(raw, 'html.parser')
print(type(soup))
```

    <class 'bs4.BeautifulSoup'>



```python
texts = []
for para in soup.find_all('p'):
    text = para.text
    texts.append(text)
print(texts[:10])
```

    ['Smog is a type of air pollutant. The word "smog" was coined in the early 20th century as a portmanteau of the words smoke and fog to refer to smoky fog.[1] The word was then intended to refer to what was sometimes known as pea soup fog, a familiar and serious problem in London from the 19th century to the mid 20th century. This kind of visible air pollution is composed of nitrogen oxides, sulfur oxides, ozone, smoke or particulates among others (less visible pollutants include carbon monoxide, CFCs and radioactive sources). Man-made smog is derived from coal emissions, vehicular emissions, industrial emissions, forest and agricultural fires and photochemical reactions of these emissions.', 'Modern smog, as found for example in Los Angeles, is a type of air pollution derived from vehicular emission from internal combustion engines and industrial fumes that react in the atmosphere with sunlight to form secondary pollutants that also combine with the primary emissions to form photochemical smog. In certain other cities, such as Delhi, smog severity is often aggravated by stubble burning in neighboring agricultural areas. The atmospheric pollution levels of Los Angeles, Beijing, Delhi, Mexico City, Tehran and other cities are increased by inversion that traps pollution close to the ground. It is usually highly toxic to humans and can cause severe sickness, shortened life or death.', '', '', 'Coinage of the term "smog" is generally attributed to Dr. Henry Antoine Des Voeux in his 1905 paper, "Fog and Smoke" for a meeting of the Public Health Congress. The July 26, 1905 edition of the London newspaper Daily Graphic quoted Des Voeux, "He said it required no science to see that there was something produced in great cities which was not found in the country, and that was smoky fog, or what was known as \'smog.\'"[2] The following day the newspaper stated that "Dr. Des Voeux did a public service in coining a new word for the London fog." However, this is predated by a Los Angeles Times article of January 19, 1893, in which the word is attributed to "a witty English writer."', "Coal fires, used to heat individual buildings or in a power-producing plant, can emit significant clouds of smoke that contributes to smog. Air pollution from this source has been reported in England since the Middle Ages.[3] London, in particular, was notorious up through the mid-20th century for its coal-caused smogs, which were nicknamed 'pea-soupers.' Air pollution of this type is still a problem in areas that generate significant smoke from burning coal, as witnessed by the 2013 autumnal smog in Harbin, China, which closed roads, schools, and the airport.", 'Traffic emissions – such as from trucks, buses, and automobiles – also contribute.[4] Airborne by-products from vehicle exhaust systems cause air pollution and are a major ingredient in the creation of smog in some large cities.[5][6][7][8]', 'The major culprits from transportation sources are carbon monoxide (CO),[9][10] nitrogen oxides (NO and NOx),[11][12][13] volatile organic compounds,[10][11] sulfur dioxide,[10] and hydrocarbons.[10] (Hydrocarbons are the main components of petroleum fuels such as gasoline and diesel fuel.) These molecules react with sunlight, heat, ammonia, moisture, and other compounds to form the noxious vapors, ground level ozone, and particles that comprise smog.[10][11]', 'Photochemical smog is the chemical reaction of sunlight, nitrogen oxides and volatile organic compounds in the atmosphere, which leaves airborne particles and ground-level ozone.[14] This noxious mixture of air pollutants may include the following:', 'A primary pollutant is an air pollutant emitted directly from a source. A secondary pollutant is not directly emitted as such, but forms when other pollutants (primary pollutants) react in the atmosphere. Examples of a secondary pollutant include ozone, which is formed when hydrocarbons (HC) and nitrogen oxides (NOx) combine in the presence of sunlight; nitrogen dioxide (NO2), which is formed as nitric oxide (NO) combines with oxygen in the air; and acid rain, which is formed when sulfur dioxide or nitrogen oxides react with water.[15] All of these harsh chemicals are usually highly reactive and oxidizing. Photochemical smog is therefore considered to be a problem of modern industrialization. It is present in all modern cities, but it is more common in cities with sunny, warm, dry climates and a large number of motor vehicles.[16] Because it travels with the wind, it can affect sparsely populated areas as well.']



```python
import re
regex = re.compile('\[[0-9]*\]')
joined_texts = '\n'.join(texts)
joined_texts = re.sub(regex, '', joined_texts)
print(type(joined_texts))
print(joined_texts[:100])
```

    <class 'str'>
    Smog is a type of air pollutant. The word "smog" was coined in the early 20th century as a portmante


In order to work on the text, the first step is to tokenise it into words.


```python
import nltk
wordlist = nltk.word_tokenize(joined_texts)
wordlist[:8]
```




    ['Smog', 'is', 'a', 'type', 'of', 'air', 'pollutant', '.']



For some other types of analysis, we'll need to create an NLTK text object


```python
good_text = nltk.Text(wordlist)
good_text.concordance('smog')
```

    Displaying 25 of 39 matches:
                                         Smog is a type of air pollutant . The wor
                                         smog '' was coined in the early 20th cent
    and radioactive sources ) . Man-made smog is derived from coal emissions , veh
    eactions of these emissions . Modern smog , as found for example in Los Angele
    mary emissions to form photochemical smog . In certain other cities , such as 
    rtain other cities , such as Delhi , smog severity is often aggravated by stub
    fe or death . Coinage of the term `` smog '' is generally attributed to Dr. He
     clouds of smoke that contributes to smog . Air pollution from this source has
     , as witnessed by the 2013 autumnal smog in Harbin , China , which closed roa
     major ingredient in the creation of smog in some large cities . The major cul
     ozone , and particles that comprise smog . Photochemical smog is the chemical
    s that comprise smog . Photochemical smog is the chemical reaction of sunlight
    active and oxidizing . Photochemical smog is therefore considered to be a prob
    automotive exhaust and photochemical smog was discovered in the 1950s by Arie 
    wo key components to the creation of smog . However , the smog created as a re
    the creation of smog . However , the smog created as a result of a volcanic er
    s been linked to the distribution of smog in some areas . For example , the cr
     has been shown to have an effect on smog distribution that is more than fossi
     than fossil fuel combustion alone . Smog is a serious problem in many cities 
    o Medical Association announced that smog is responsible for an estimated 9,50
     who had healthy babies , found that smog in the San Joaquin Valley area of Ca
    w the current accepted safe levels . Smog can form in almost any climate where
    hi 's children and women . The dense smog in Delhi during winter season result
    results in severe intensification of smog over Delhi . The state government of
    ust our iron . '' Severe episodes of smog continued in the 19th and 20th centu


And once we've done all that work creating clean text, it's a good idea to save it for later.


```python
%cd
! mkdir smog
%cd smog
```

    /home/researcher
    mkdir: cannot create directory 'smog': File exists
    /home/researcher/smog



```python
NLTK_file = open("NLTK-Smog.txt", "w", encoding='UTF-8')
NLTK_file.write(str(wordlist))
NLTK_file.close()
```


```python
text_file = open("Smog-text.txt", "w", encoding='UTF-8')
text_file.write(joined_texts)
text_file.close()
```


```python
joined_texts[2450:2471]
```




    'of this type is still'




```python
#joined_texts[2450:2470]
text_file = open("Smog-text.txt", "w", encoding='UTF-8')
text_file.write(joined_texts)
text_file.close()
```

Now have a look at the two files you've created in the file management system. Open them. How is the nltk file different from the .txt file?

**Challenge!**
* Find a webpage of interest to your studies and use Beautiful Soup to extract the text
* Tokenise the text
* Find the most common words in your text (Extension: remove the stop words)
* Find trigrams in your text 
* Save your text to a text file

*Hint*: feel free to collude with your neighbours and please copy and paste our previous code! Copying and pasting are essential skills of developers, as well as googling error messages (seriously!). If you don't believe me, ask a computer scientist. 


```python

```
