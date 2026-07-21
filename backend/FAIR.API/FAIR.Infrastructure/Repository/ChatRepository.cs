using FAIR.Domain.Entities.Chat;
using FAIR.Domain.Interfaces;
using FAIR.Infrastructure.Data;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FAIR.Infrastructure.Repository
{
    public class ChatRepository(AppDbContext context) : IChatRepository
    {
        private readonly AppDbContext _context = context;
        public async Task<Message> SaveMessageAsync(Message message)
        {
            var sender = await _context.Users.FindAsync(message.SenderId);
            if (sender != null)
            {
                message.SenderName = sender.UserName!;
            }

            _context.Messages.Add(message);
            await _context.SaveChangesAsync();
            return message;
        }
        public async Task<List<Message>> GetPrivateMessagesAsync(string userId1, string userId2)
        {
            return await _context.Messages
                .Where(m => m.ReceiverId != null &&
                          ((m.SenderId == userId1 && m.ReceiverId == userId2) ||
                           (m.SenderId == userId2 && m.ReceiverId == userId1)))
                .OrderBy(m => m.CreateData)
                .ToListAsync();
        }
    }
}
