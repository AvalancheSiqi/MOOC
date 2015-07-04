//
//  MemeViewController.swift
//  MemeMe
//
//  Created by SiqiWu on 28/06/2015.
//  Copyright (c) 2015 Siqi. All rights reserved.
//

import UIKit

class MemeViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate, UITextFieldDelegate {
    
    @IBOutlet weak var imagePickerView: UIImageView!
    @IBOutlet weak var top: UITextField!
    @IBOutlet weak var bottom: UITextField!
    @IBOutlet weak var navbar: UINavigationBar!
    @IBOutlet weak var toolbar: UIToolbar!
    @IBOutlet weak var shareButton: UIBarButtonItem!
    @IBOutlet weak var cameraButton: UIBarButtonItem!
    @IBOutlet weak var cancleButton: UIBarButtonItem!
    
    
    // - Set initial textfield and placeholder
    let memeTextAttributes = [
        NSStrokeColorAttributeName: UIColor.blackColor(),
        NSForegroundColorAttributeName: UIColor.whiteColor(),
        NSFontAttributeName: UIFont(name: "HelveticaNeue-CondensedBlack", size: 40)!,
        NSStrokeWidthAttributeName: -3.0,
        ]

    private func getInitialText(textField: UITextField) -> String {
        switch textField {
        case top:
            return "TOP"
        case bottom:
            return "BOTTOM"
        default:
            return "NO VALID"
        }
    }
    
    private func setInitialTextField(textField: UITextField) {
        textField.text = self.getInitialText(textField)
        textField.defaultTextAttributes = memeTextAttributes
        textField.textAlignment = NSTextAlignment.Center
        textField.delegate = self
    }
    
    override func prefersStatusBarHidden() -> Bool {
        return true
    }
    
    
    // - Life cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        self.setInitialTextField(top)
        self.setInitialTextField(bottom)
    }
    
    override func viewWillAppear(animated: Bool) {
        super.viewWillAppear(animated)
        top.hidden = false
        bottom.hidden = false
        shareButton.enabled = imagePickerView.image != nil
        cameraButton.enabled = UIImagePickerController.isSourceTypeAvailable(UIImagePickerControllerSourceType.Camera)
        self.subscribeToKeyboardNotifications()
        
    }

    override func viewWillDisappear(animated: Bool) {
        super.viewWillDisappear(animated)
        self.unsubscribeToKeyboardNotifications()
    }
    
    @IBAction func cancel(sender: AnyObject) {
        self.dismissViewControllerAnimated(true, completion: nil)
    }
    
    
    // - Pick an image either from album or camera
    private func pickAnImage(sourceType: UIImagePickerControllerSourceType) {
        let imagePicker = UIImagePickerController()
        imagePicker.delegate = self
        imagePicker.sourceType = sourceType
        self.presentViewController(imagePicker, animated: true, completion: nil)
    }
    
    @IBAction func pickAnImageFromAlbum(sender: AnyObject) {
        pickAnImage(.PhotoLibrary)
    }
    
    @IBAction func pickAnImageFromCamera(sender: AnyObject) {
        pickAnImage(.Camera)
    }
    
    
    // - Implement UIImagePickerControllerDelegate
    func imagePickerController(picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [NSObject : AnyObject]) {
        if let chosenImage = info[UIImagePickerControllerOriginalImage] as? UIImage {
            self.imagePickerView.contentMode = .ScaleAspectFit
            self.imagePickerView.image = chosenImage
        }
        self.dismissViewControllerAnimated(true, completion: nil)
    }
    
    func imagePickerControllerDidCancel(picker: UIImagePickerController) {
        self.dismissViewControllerAnimated(true, completion: nil)
    }
    
    
    // - Implement UITextViewDelegate
    func textFieldShouldBeginEditing(textField: UITextField) -> Bool {
        if textField.text == self.getInitialText(textField) {
            textField.text = ""
        }
        return true
    }
    
    func textFieldShouldReturn(textField: UITextField) -> Bool {
        NSNotificationCenter.defaultCenter().addObserver(self, selector: "keyboardWillHide:", name: UIKeyboardWillHideNotification, object: nil)
        
        textField.resignFirstResponder()
        return true
    }
    
    
    // - Implement keyboard notification
    func subscribeToKeyboardNotifications() {
        NSNotificationCenter.defaultCenter().addObserver(self, selector: "keyboardWillShow:", name: UIKeyboardWillShowNotification, object: nil)
    }
    
    func unsubscribeToKeyboardNotifications() {
        NSNotificationCenter.defaultCenter().removeObserver(self)
    }
    
    func keyboardWillShow(notification: NSNotification) {
        if self.bottom.editing {
            self.view.frame.origin.y -= getKeyboardHeight(notification)
        }
    }
    
    func keyboardWillHide(notification: NSNotification) {
        self.view.frame.origin.y = 0
    }
    
    func getKeyboardHeight(notification: NSNotification) -> CGFloat {
        let userInfo = notification.userInfo
        let keyboardSize = userInfo![UIKeyboardFrameEndUserInfoKey] as! NSValue
        return keyboardSize.CGRectValue().height
    }
    
    
    // - Share and save memed image
    @IBAction func shareAndSaveMeme(sender: AnyObject) {
        var memedImage = generateMemedImage()
        let activityVC = UIActivityViewController(activityItems: [memedImage], applicationActivities: nil)
        
        func saveMemeAfterSharing(activity: String!, success: Bool, items: [AnyObject]!, error: NSError!) {
            if (success) {
                self.save()
                self.dismissViewControllerAnimated(true, completion: nil)
            }
        }
        activityVC.completionWithItemsHandler = saveMemeAfterSharing
        self.presentViewController(activityVC, animated: true, completion: nil)
    }
    
    func save() {
        // Create the meme
        var meme = Meme(topText: top.text, bottomText: bottom.text, originalImage: imagePickerView.image!, memedImage: generateMemedImage())
        (UIApplication.sharedApplication().delegate as! AppDelegate).memes.append(meme)
    }
    
    func generateMemedImage() -> UIImage
    {
        // Hide toolbar and navbar
        self.navbar.hidden = true
        self.toolbar.hidden = true
        
        // Render view to an image
        UIGraphicsBeginImageContext(self.view.frame.size)
        self.view.drawViewHierarchyInRect(self.view.frame, afterScreenUpdates: true)
        let memedImage : UIImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        // Show toolbar and navbar
        self.navbar.hidden = false
        self.toolbar.hidden = false
        
        return memedImage
    }
    
}

