//
//  HistoryViewController.swift
//  Roshambo
//
//  Created by SiqiWu on 30/06/2015.
//  Copyright (c) 2015 Siqi. All rights reserved.
//

import UIKit

class HistoryViewController: UIViewController, UITableViewDataSource {
    
    var history: [RPSMatch]!
    
    override func viewWillAppear(animated: Bool) {
        super.viewWillAppear(animated)
    }
    
    func tableView(tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return self.history.count
    }
    
    func tableView(tableView: UITableView, cellForRowAtIndexPath indexPath: NSIndexPath) -> UITableViewCell {
        let CellID = "ResultCell"
        let cell = tableView.dequeueReusableCellWithIdentifier(CellID) as! UITableViewCell
        let match = self.history[indexPath.row]
        
        cell.textLabel?.text = match.resultTitle
        cell.detailTextLabel?.text = match.resultDescription
        
        return cell
    }

    @IBAction func dismissHistory() {
        self.dismissViewControllerAnimated(true, completion: nil)
    }
}
