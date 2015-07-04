//
//  MemeTableViewController.swift
//  MemeMe
//
//  Created by SiqiWu on 2/07/2015.
//  Copyright (c) 2015 Siqi. All rights reserved.
//

import UIKit

class MemeTableViewController: UIViewController, UITableViewDelegate, UITableViewDataSource {
    
    var memes: [Meme]!
    @IBOutlet var tableView: UITableView!
    
    
    // - LifeCycle
    override func viewWillAppear(animated: Bool) {
        super.viewWillAppear(animated)
        tableView.delegate = self
        tableView.dataSource = self
        
        let appDelegate = (UIApplication.sharedApplication().delegate as! AppDelegate)
        memes = appDelegate.memes
        
        self.tableView.reloadData()
    }
    
    
    // - TableView DataSource and Delegate
    func tableView(tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return self.memes.count
    }
    
    func tableView(tableView: UITableView, cellForRowAtIndexPath indexPath: NSIndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCellWithIdentifier("MemeTableViewCell") as! MemeTableViewCell
        let meme = self.memes[indexPath.row]
        
        cell.topText.text = meme.topText
        cell.bottomText.text = meme.bottomText
        cell.originalImage.image = meme.originalImage
        return cell
    }
    
    func tableView(tableView: UITableView, didSelectRowAtIndexPath indexPath: NSIndexPath) {
        let detailVC = self.storyboard!.instantiateViewControllerWithIdentifier("MemeDetailViewController") as! MemeDetailViewController
        
        detailVC.memedImage = self.memes[indexPath.row].memedImage
        detailVC.hidesBottomBarWhenPushed = true
        self.navigationController!.pushViewController(detailVC, animated: true)
    }
    
    
    // - Add new meme
    @IBAction func addMeme(sender: AnyObject) {
        let memeVC = self.storyboard!.instantiateViewControllerWithIdentifier("MemeViewController") as! MemeViewController
        
        self.presentViewController(memeVC, animated: true, completion: nil)
    }
    
}

