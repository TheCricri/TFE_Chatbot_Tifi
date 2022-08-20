"""
TFE - Chatbot Tifi - Technifutur
by Nicolas Christiaens
"""

from tkinter import *
import PIL.Image
import PIL.ImageTk
from concurrent import futures

# Create the class for our basic interface
class Interface:
    def __init__(self,MyChatbot):
        self.MyChatbot = MyChatbot
        self.thread_pool = futures.ThreadPoolExecutor(max_workers=1)
        
        # Create the main Window
        self.Window = Tk()
        self.Window.title("Chatbot")
        self.Window.resizable(width = False,
                              height = False)
        max_width = self.Window.winfo_screenwidth()
        max_height = self.Window.winfo_screenheight()
        main_width = max_width*0.4
        main_height = max_height*0.7
        ratio = int((max_height+max_width)/200)
        self.Window.configure(width = main_width,
                              height = main_height,
                              borderwidth=0,
                              bg = "blue")
        
        # Create the 'top Icon' section
        self.top_rec = Canvas(self.Window,
                              bg="white",highlightbackground="#ABB2B9",
                              highlightthickness=2)
        self.top_rec.place(relwidth = 1,relheight=0.1)
        im = PIL.Image.open("Images/Logo.png")
        im = im.resize((int(main_width*0.15),int(main_height*0.09)), PIL.Image.ANTIALIAS)
        icon1 = PIL.ImageTk.PhotoImage(im)
        self.top_rec.create_image(main_width*0.14,main_height*0.05, image=icon1)
        self.top_rec.create_text(main_width*0.5,
                                 main_height*0.05,
                                 fill="black",
                                 font="Helvetica {} bold".format(str(ratio+2)),
                                 text=self.MyChatbot.getName())
         
        # Create the 'chat field' section
        self.chatField = Text(self.Window,
                              bg = "lightblue",
                              highlightthickness=4,
                              highlightbackground="#ABB2B9",
                              wrap=WORD,
                              fg = "black",
                              font = "Helvetica {}".format(str(ratio+1)),
                              padx = 20,
                              pady = 6)
        self.chatField.place(relheight = 0.75,
                            relwidth = 1,
                            rely = 0.1)
        self.chatField.config(cursor = "arrow",
                             insertbackground="blue",
                             state = DISABLED)
        scrollbar = Scrollbar(self.chatField)
        scrollbar.place(relheight = 1,relx = 1)
        scrollbar.config(command = self.chatField.yview)
        
        # Create the 'send message' section
        self.bot_label = Label(self.Window,
                                  bg = "white")
        self.bot_label.place(relwidth = 1,
                               relheight = 0.25,
                               rely = 0.85)
        
        # Create the entry for the message
        self.entryMsg = Entry(self.bot_label,
                              bg = "#ABB2B9",
                              fg = "black",
                              font = "Helvetica {}".format(str(ratio)))
        self.entryMsg.place(relwidth=0.79,
                            relheight=0.5,
                            rely=0.035,
                            relx=0)
        self.entryMsg.focus()

        # Create the button for sending the message
        im2 = PIL.Image.open("Images/send.png")
        im2 = im2.resize((int(main_width*0.05),int(main_height*0.05)), PIL.Image.ANTIALIAS)
        icon2 = PIL.ImageTk.PhotoImage(im2)
        self.buttonSend = Button(self.bot_label,
                                borderwidth = 0,
                                image = icon2,
                                bg = "white",
                                command = lambda : self.activationHandler())
        self.buttonSend.place(relx = 0.80,
                              rely = 0.035,
                              relheight = 0.5,
                              relwidth = 0.2)
        
        # Bind return to activate the button
        self.Window.bind('<Return>', self.ReturnActivate)
        
        # Place Window on the top on the stack
        self.Window.lift()
        
        # Insert the welcome message
        self.welcomeMessage()
        
        # Create the reset button
        im3 = PIL.Image.open("Images/reset.png")
        im3 = im3.resize((int(main_width*0.05),int(main_height*0.05)), PIL.Image.ANTIALIAS)
        icon3 = PIL.ImageTk.PhotoImage(im3)
        self.buttonReset = Button(self.top_rec,
                                borderwidth = 0,
                                image = icon3,
                                bg = "white",
                                activebackground = "white",
                                command = lambda : self.resetHandler())
        self.buttonReset.place(relx = 0.9,
                              rely = 0.25,
                              relheight = 0.5,
                              relwidth = 0.075)
        
        # Start the loop
        self.Window.mainloop()
        
    # Function to handle the welcome message
    def welcomeMessage(self):
        text = f"{self.MyChatbot.getName()}: {self.MyChatbot.getWelcomeMessage()}\n\n"
        self.chatField.configure(state=NORMAL)
        self.chatField.insert(END,text)
        self.chatField.configure(state=DISABLED)
        
        self.chatField.see(END)
 
    # Function to handle the activation of the "Send" button
    def activationHandler(self):
        # Get the message
        tmp_msg = self.entryMsg.get()
        
        # Skip if no message
        if not tmp_msg:
            return
        
        # Delete the entry
        self.entryMsg.delete(0,END)
        
        # Stop entry
        self.entryMsg.configure(state=DISABLED)
        
        # Get the message
        self.current_msg = tmp_msg
        
        # Handle thread
        self.thread_pool.submit(self.activateButtonSend)
        
        # Stop entry
        self.entryMsg.configure(state=NORMAL)
 
    # Function to handle the activation of the buttonSend
    def activateButtonSend(self):
        # Print the user message
        text = f"You: {self.current_msg}\n\n"
        self.chatField.configure(state=NORMAL)
        self.chatField.insert(END,text)
        self.chatField.configure(state=DISABLED)
        
        # Print the chatbot response
        msg = self.MyChatbot.getResponse(self.current_msg)
        text = f"{self.MyChatbot.getName()}: {msg}\n\n"
        self.chatField.configure(state=NORMAL)
        self.chatField.insert(END,text)
        self.chatField.configure(state=DISABLED)
        self.chatField.see(END)
        
    # Function to handle the return on keyboard
    def ReturnActivate(self,event):
        self.activationHandler()
    
    # Function to handle the reset button
    def resetHandler(self):
        self.chatField.configure(state=NORMAL)
        self.chatField.delete('1.0',END)
        self.chatField.configure(state=DISABLED)
        self.MyChatbot.resetState()
        self.welcomeMessage()